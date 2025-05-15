import code
import torch
import torchlens
from app.core.graph_parser import parse_torchlens_object
from app.ui.graph_visualizer import launch_ui
import ast
import types

# Context storage for analysis results and hooks
current_analysis_context = {
    'original_model': None,
    'lens_object': None,
    'parsed_nodes': None,
    'parsed_edges': None,
    'active_hooks': {}  # To store hook handles, {node_uid: handle}
}

# Custom banner for the REPL
BANNER = """
Welcome to the PyTorch Interactive REPL!
Available functions:
  - torch: The PyTorch module.
  - analyze_model(model, dummy_input_shape=None): Analyzes a model and launches the UI.
  - attach_hook(node_uid, hook_function_str, hook_type='forward'): Attaches a hook to a module.
Type exit() or Ctrl-D to exit.
"""

def analyze_model(model, dummy_input_shape=None):
    if model is None:
        print("Error: Model is None. Please provide a valid PyTorch model.")
        return None

    if not isinstance(model, torch.nn.Module):
        print("Error: The provided object is not a PyTorch nn.Module.")
        return None

    X = None
    if dummy_input_shape:
        try:
            X = torch.randn(dummy_input_shape)
        except Exception as e:
            print(f"Error creating dummy input with shape {dummy_input_shape}: {e}")
            return None
    else:
        try:
            X = torch.randn(1, 3, 224, 224) # Default for many image models
            print(f"Used default dummy input shape: (1, 3, 224, 224)")
        except Exception as e:
            print(f"Error creating default dummy input: {e}")
            print("Please provide a dummy_input_shape argument, e.g., analyze_model(your_model, dummy_input_shape=(1, 1, 28, 28))")
            return None
    
    print(f"Analyzing model with input shape: {X.shape}")
    try:
        lens = torchlens.log_model(model, X)
        print("Model analysis complete. Lens object created.")
        
        if lens:
            parsed_nodes, parsed_edges = parse_torchlens_object(lens)
            
            # Store analysis results in the context
            current_analysis_context['original_model'] = model
            current_analysis_context['lens_object'] = lens
            current_analysis_context['parsed_nodes'] = parsed_nodes
            current_analysis_context['parsed_edges'] = parsed_edges
            current_analysis_context['active_hooks'] = {} # Reset hooks for new model

            print("\n--- Parsed Graph Information (Summary) ---")
            print(f"Found {len(parsed_nodes)} nodes and {len(parsed_edges)} edges.")
            print("--- End Parsed Graph Information ---")

            print("\nLaunching UI with graph data...")
            launch_ui(parsed_nodes, parsed_edges, hook_callback=attach_hook) # This blocks
            print("UI closed.")
            return "Analysis complete and UI launched."
        else:
            print("torchlens.log_model returned None.")
            return "Analysis failed: torchlens.log_model returned None."
            
    except Exception as e:
        print(f"Error during torchlens.log_model, parsing, or UI launch: {e}")
        return f"Analysis failed: {e}"

def attach_hook(node_uid, hook_function_str, hook_type='forward'):
    if not current_analysis_context['original_model'] or not current_analysis_context['lens_object']:
        print("Error: No model has been analyzed yet. Please run analyze_model() first.")
        return False

    target_module_name = None
    # target_layer_node_obj = None # Not strictly needed outside this initial search

    # Find the full name of the module from lens_object.layer_nodes first
    if current_analysis_context['lens_object'].layer_nodes:
        for layer_node in current_analysis_context['lens_object'].layer_nodes:
            if hasattr(layer_node, 'uid') and layer_node.uid == node_uid:
                # target_layer_node_obj = layer_node # Found the lens object node
                if hasattr(layer_node, 'full_name'):
                    target_module_name = layer_node.full_name
                else: # Fallback if full_name isn't directly there
                    target_module_name = getattr(layer_node, 'name', None)
                break # Found the node in lens_object, proceed with this name
    
    # Fallback to parsed_nodes if not found or if target_module_name is still None
    if not target_module_name and current_analysis_context['parsed_nodes']:
         for p_node in current_analysis_context['parsed_nodes']:
             if p_node['id'] == node_uid:
                 # Prefer 'full_name' from parsed_nodes if available, else 'label'
                 target_module_name = p_node.get('full_name', p_node.get('label'))
                 break

    if not target_module_name:
        print(f"Error: Could not find module name for node_uid '{node_uid}'.")
        return False

    try:
        module_instance = current_analysis_context['original_model']
        # Attempt to use get_submodule first as it's safer for nested paths
        try:
            module_instance = module_instance.get_submodule(target_module_name)
        except AttributeError: # Fallback for older PyTorch or if get_submodule doesn't exist/work
            print(f"Model does not have get_submodule or failed for '{target_module_name}'. Trying sequential getattr...")
            if '.' in target_module_name:
                for part in target_module_name.split('.'):
                    module_instance = getattr(module_instance, part)
            elif target_module_name: # For top-level modules (module_name is not empty)
                module_instance = getattr(module_instance, target_module_name)
            else: # target_module_name was empty or somehow invalid
                print(f"Error: Module name '{target_module_name}' is invalid for sequential getattr.")
                return False
        except Exception as e_get_submodule: # Catch other errors from get_submodule
            print(f"Error during get_submodule for '{target_module_name}': {e_get_submodule}")
            return False


    except AttributeError:
        print(f"Error: Could not retrieve submodule '{target_module_name}' from the model using any method.")
        return False
    except Exception as e_retrieve: # Catch any other unexpected error during retrieval
        print(f"An unexpected error occurred retrieving module '{target_module_name}': {e_retrieve}")
        return False


    if not isinstance(module_instance, torch.nn.Module):
         print(f"Error: Retrieved object for '{target_module_name}' is not an nn.Module. Type: {type(module_instance)}")
         return False

    hook_fn = None
    try:
        temp_scope = {'torch': torch, 'nn': torch.nn}
        if not hook_function_str.strip().startswith('def '):
            print("Error: hook_function_str must be a full function definition (e.g., 'def my_hook(...): ...').")
            return False

        tree = ast.parse(hook_function_str)
        func_name = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
        
        if not func_name:
            print("Error: Could not determine function name from hook_function_str.")
            return False

        exec(hook_function_str, temp_scope)
        hook_fn = temp_scope[func_name]

        if not isinstance(hook_fn, types.FunctionType):
            print(f"Error: Compiled object '{func_name}' is not a function.")
            return False

    except Exception as e:
        print(f"Error compiling or executing hook_function_str: {e}")
        return False

    handle = None
    try:
        # Remove previous hook for this node if any, before adding new one
        if node_uid in current_analysis_context['active_hooks']:
             try:
                 current_analysis_context['active_hooks'][node_uid].remove()
                 print(f"Removed existing hook for node {node_uid}.")
             except Exception as e_remove:
                 print(f"Note: Could not remove previous hook for node {node_uid}: {e_remove}")
        
        if hook_type == 'forward':
            handle = module_instance.register_forward_hook(hook_fn)
            print(f"Forward hook '{func_name}' attached to module '{target_module_name}' (UID: {node_uid}).")
        elif hook_type == 'backward':
            handle = module_instance.register_full_backward_hook(hook_fn) # For nn.Module
            print(f"Full backward hook '{func_name}' attached to module '{target_module_name}' (UID: {node_uid}).")
        else:
            print(f"Error: Invalid hook_type '{hook_type}'. Use 'forward' or 'backward'.")
            return False
        
        current_analysis_context['active_hooks'][node_uid] = handle
        print(f"Successfully attached {hook_type} hook '{func_name}' to module '{target_module_name}' (UID: {node_uid}).")
        return True

    except Exception as e:
        print(f"Error attaching hook to module '{target_module_name}': {e}")
        return False


# Define console locals, including the new attach_hook function
console_locals = {
    "torch": torch,
    "analyze_model": analyze_model,
    "attach_hook": attach_hook,
    # Potentially add current_analysis_context for advanced debugging if needed,
    # but it's generally better to interact via functions.
    # "analysis_context": current_analysis_context 
}

# Instantiate the console
console = code.InteractiveConsole(locals=console_locals)

# Start the REPL with the custom banner
console.interact(banner=BANNER)
