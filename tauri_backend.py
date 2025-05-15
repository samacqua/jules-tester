import json
import sys
import importlib.util
import torch
import torch.nn # Explicitly import nn for hook function scope
import torchlens # Assuming torchlens is installed
import ast
import types
from app.core.graph_parser import parse_torchlens_object

# Global context (kept minimal for stateless sidecar calls, reloaded per call)
# If we were to make it stateful (sidecar runs continuously):
# current_model_context = {
#     'original_model': None,
#     'lens_object': None,
# }

def _load_model_from_path(model_file_path, model_class_name):
    """Loads a PyTorch model class from a given file path and class name."""
    try:
        spec = importlib.util.spec_from_file_location("user_model_module", model_file_path)
        if spec is None:
            return None, f"Could not create module spec from path: {model_file_path}"
        
        user_model_module = importlib.util.module_from_spec(spec)
        sys.modules["user_model_module"] = user_model_module # Add to sys.modules for unpickling if model uses it
        spec.loader.exec_module(user_model_module)
        
        model_class = getattr(user_model_module, model_class_name)
        return model_class, None
    except FileNotFoundError:
        return None, f"Model file not found: {model_file_path}"
    except AttributeError:
        return None, f"Class '{model_class_name}' not found in file '{model_file_path}'"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def handle_analyze_model(payload):
    try:
        model_file_path = payload.get('model_file_path')
        model_class_name = payload.get('model_class_name')
        dummy_input_shape = payload.get('dummy_input_shape')

        if not all([model_file_path, model_class_name, dummy_input_shape]):
            return {"status": "error", "message": "Missing parameters for analyze_model"}

        model_class, error_msg = _load_model_from_path(model_file_path, model_class_name)
        if error_msg:
            return {"status": "error", "message": error_msg}

        model_instance = model_class() # Assumes model can be instantiated without args

        X = torch.randn(dummy_input_shape)
        
        lens = torchlens.log_model(model_instance, X)
        nodes, edges = parse_torchlens_object(lens)
        
        # Store for potential immediate follow-up hook attachment if we make sidecar stateful later
        # For now, lens and model_instance are local to this call.
        # current_model_context['original_model'] = model_instance 
        # current_model_context['lens_object'] = lens

        return {"status": "success", "data": {"nodes": nodes, "edges": edges}}

    except Exception as e:
        return {"status": "error", "message": f"Error in analyze_model: {str(e)}"}

def handle_attach_hook(payload):
    try:
        model_file_path = payload.get('model_file_path')
        model_class_name = payload.get('model_class_name')
        dummy_input_shape = payload.get('dummy_input_shape') # Needed to re-create model and lens
        node_uid = payload.get('node_uid')
        hook_function_str = payload.get('hook_function_str')
        hook_type = payload.get('hook_type', 'forward')

        if not all([model_file_path, model_class_name, dummy_input_shape, node_uid, hook_function_str]):
            return {"status": "error", "message": "Missing parameters for attach_hook"}

        model_class, error_msg = _load_model_from_path(model_file_path, model_class_name)
        if error_msg:
            return {"status": "error", "message": error_msg}
        
        model_instance = model_class() # Re-instantiate the model

        # We need the Lens object to map node_uid to module_full_name
        # This means re-running torchlens.log_model or having it passed.
        # For simplicity of stateless sidecar, re-run analysis part.
        temp_X = torch.randn(dummy_input_shape)
        lens_object = torchlens.log_model(model_instance, temp_X)

        if not lens_object or not hasattr(lens_object, 'layer_nodes'):
             return {"status": "error", "message": "Failed to get Lens object or layer_nodes for hook attachment."}


        target_module_name = None
        for layer_node in lens_object.layer_nodes:
            if hasattr(layer_node, 'uid') and layer_node.uid == node_uid:
                target_module_name = getattr(layer_node, 'full_name', getattr(layer_node, 'name', None))
                break
        
        if not target_module_name:
            return {"status": "error", "message": f"Could not find module name for node_uid '{node_uid}'"}

        actual_module_instance = model_instance
        try:
            actual_module_instance = model_instance.get_submodule(target_module_name)
        except AttributeError: # Fallback for older PyTorch or if name isn't a path
            try:
                for part in target_module_name.split('.'):
                    actual_module_instance = getattr(actual_module_instance, part)
            except AttributeError:
                return {"status": "error", "message": f"Could not retrieve submodule '{target_module_name}' from the model."}
        
        if not isinstance(actual_module_instance, torch.nn.Module):
             return {"status": "error", "message": f"Retrieved object for '{target_module_name}' is not an nn.Module."}


        hook_fn = None
        temp_scope = {'torch': torch, 'nn': torch.nn}
        if not hook_function_str.strip().startswith('def '):
            return {"status": "error", "message": "hook_function_str must be a full function definition."}

        tree = ast.parse(hook_function_str)
        func_name = next((node.name for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        
        if not func_name:
            return {"status": "error", "message": "Could not determine function name from hook_function_str."}

        exec(hook_function_str, temp_scope)
        hook_fn = temp_scope[func_name]

        if not isinstance(hook_fn, types.FunctionType):
            return {"status": "error", "message": f"Compiled object '{func_name}' is not a function."}

        if hook_type == 'forward':
            actual_module_instance.register_forward_hook(hook_fn)
        elif hook_type == 'backward':
            actual_module_instance.register_full_backward_hook(hook_fn)
        else:
            return {"status": "error", "message": f"Invalid hook_type '{hook_type}'. Use 'forward' or 'backward'."}
        
        # Note: The hook is registered on this instance of model_instance.
        # This instance is not persisted by the sidecar after this call.
        # The frontend needs to be aware that hooks are transient with this backend model.

        return {"status": "success", "message": f"Hook '{func_name}' attached to module '{target_module_name}' (UID: {node_uid})."}

    except Exception as e:
        return {"status": "error", "message": f"Error in attach_hook: {str(e)}"}

def main():
    try:
        raw_payload = sys.stdin.readline()
        if not raw_payload:
            # Handle empty input, perhaps from a test or incorrect invocation
            print(json.dumps({"status": "error", "message": "No input received"}), file=sys.stdout, flush=True)
            return

        payload = json.loads(raw_payload)
        command = payload.get("command")
        response = None

        if command == "analyze_model":
            response = handle_analyze_model(payload)
        elif command == "attach_hook":
            response = handle_attach_hook(payload)
        # Can add an 'initialize' command later if needed
        # elif command == "initialize":
        #     response = {"status": "success", "message": "Backend initialized"}
        else:
            response = {"status": "error", "message": f"Unknown command: {command}"}
        
        print(json.dumps(response), file=sys.stdout, flush=True)

    except json.JSONDecodeError:
        print(json.dumps({"status": "error", "message": "Invalid JSON input"}), file=sys.stdout, flush=True)
    except Exception as e:
        # Catch-all for unexpected errors during main execution
        print(json.dumps({"status": "error", "message": f"Unhandled backend error: {str(e)}"}), file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
