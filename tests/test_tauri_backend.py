import unittest
import json
import io
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to sys.path to allow importing tauri_backend and app.core
# This is often needed if tests are run from the 'tests' directory or via 'discover'
# and the module tauri_backend is in the root.
# Adjust if your test runner handles this differently.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.insert(0, project_root)

# Now we can import tauri_backend
# However, tauri_backend.py itself is a script. To test its functions,
# it's better if its core logic (handlers, main) can be imported without side effects.
# For this test, we'll primarily test it by invoking its main() through subprocess or by refactoring.
# Given the current structure, we'll patch stdin/stdout and call main().

# If tauri_backend.py is in root, and tests are in tests/, this import might fail
# depending on how tests are run. Assuming PYTHONPATH or cwd is project root.
import tauri_backend 

# For mocking torch and torchlens if they are not installed in test environment
# or to speed up tests. For now, assume they are available as per requirements.txt.

class TestTauriBackend(unittest.TestCase):

    def setUp(self):
        self.original_stdin = sys.stdin
        self.original_stdout = sys.stdout
        # Correct path assuming tests/ is one level down from project root
        self.sample_model_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'sample_model.py')
        # Ensure sample_model_path is absolute for importlib
        self.sample_model_path = os.path.abspath(self.sample_model_path)


    def tearDown(self):
        sys.stdin = self.original_stdin
        sys.stdout = self.original_stdout

    def _run_main_with_input(self, input_json):
        sys.stdin = io.StringIO(json.dumps(input_json) + '\n')
        sys.stdout = io.StringIO()
        
        # This relies on tauri_backend.main() being callable.
        # If tauri_backend.py is purely a script, this might need adjustment
        # or use subprocess.
        try:
            tauri_backend.main() # Call the main function from the imported module
        except SystemExit: # Allow SystemExit if main() calls it
            pass 
            
        output_str = sys.stdout.getvalue()
        try:
            return json.loads(output_str)
        except json.JSONDecodeError:
            self.fail(f"Backend did not produce valid JSON output. Output: {output_str}")

    def test_invalid_json_input(self):
        sys.stdin = io.StringIO("this is not json\n")
        sys.stdout = io.StringIO()
        tauri_backend.main()
        result = json.loads(sys.stdout.getvalue())
        self.assertEqual(result.get("status"), "error")
        self.assertIn("Invalid JSON input", result.get("message", ""))

    def test_empty_input(self):
        sys.stdin = io.StringIO("\n") # Empty line
        sys.stdout = io.StringIO()
        tauri_backend.main()
        result = json.loads(sys.stdout.getvalue())
        self.assertEqual(result.get("status"), "error")
        # Check if result is not None before accessing "message"
        self.assertIn("No input received", result.get("message", "") if result else "No result")


    def test_unknown_command(self):
        payload = {"command": "non_existent_command"}
        result = self._run_main_with_input(payload)
        self.assertEqual(result.get("status"), "error")
        self.assertIn("Unknown command", result.get("message", ""))

    def test_analyze_model_success(self):
        payload = {
            "command": "analyze_model",
            "model_file_path": self.sample_model_path,
            "model_class_name": "SimpleNN", # From app/sample_model.py
            "dummy_input_shape": [1, 10] # SimpleNN expects (batch_size, 10)
        }
        result = self._run_main_with_input(payload)
        self.assertEqual(result.get("status"), "success", msg=f"Analyze model failed: {result.get('message')}")
        self.assertIn("data", result)
        self.assertIn("nodes", result["data"])
        self.assertIn("edges", result["data"])
        self.assertTrue(len(result["data"]["nodes"]) > 0) # SimpleNN should have nodes

    def test_analyze_model_missing_params(self):
        payload = {"command": "analyze_model", "model_file_path": self.sample_model_path}
        result = self._run_main_with_input(payload)
        self.assertEqual(result.get("status"), "error")
        self.assertIn("Missing parameters", result.get("message", ""))

    def test_analyze_model_file_not_found(self):
        payload = {
            "command": "analyze_model",
            "model_file_path": "non_existent_model.py",
            "model_class_name": "SomeModel",
            "dummy_input_shape": [1, 3, 32, 32]
        }
        result = self._run_main_with_input(payload)
        self.assertEqual(result.get("status"), "error")
        self.assertIn("Model file not found", result.get("message", ""))
        
    def test_analyze_model_class_not_found(self):
        payload = {
            "command": "analyze_model",
            "model_file_path": self.sample_model_path,
            "model_class_name": "NonExistentClass",
            "dummy_input_shape": [1,10]
        }
        result = self._run_main_with_input(payload)
        self.assertEqual(result.get("status"), "error")
        self.assertIn("NonExistentClass' not found", result.get("message", ""))


    def test_attach_hook_success(self):
        mock_lens_node = MagicMock()
        mock_lens_node.uid = "test_uid_layer1" # Corresponds to SimpleNN's layer1
        mock_lens_node.full_name = "layer1"    # Actual attribute name in SimpleNN
        
        mock_lens_object = MagicMock()
        mock_lens_object.layer_nodes = [mock_lens_node]

        with patch('tauri_backend.torchlens.log_model', return_value=mock_lens_object):
            payload = {
                "command": "attach_hook",
                "model_file_path": self.sample_model_path,
                "model_class_name": "SimpleNN",
                "dummy_input_shape": [1, 10], # Consistent with SimpleNN
                "node_uid": "test_uid_layer1",   # UID from our mock
                "hook_function_str": "def my_hook(module, input, output): return output",
                "hook_type": "forward"
            }
            result = self._run_main_with_input(payload)
            self.assertEqual(result.get("status"), "success", msg=f"Attach hook failed: {result.get('message')}")
            self.assertIn("Hook 'my_hook' attached", result.get("message", ""))


    def test_attach_hook_missing_params(self):
        payload = {"command": "attach_hook", "node_uid": "some_uid"}
        result = self._run_main_with_input(payload)
        self.assertEqual(result.get("status"), "error")
        self.assertIn("Missing parameters", result.get("message", ""))

if __name__ == '__main__':
    unittest.main()
