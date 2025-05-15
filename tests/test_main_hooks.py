import unittest
import torch
import torch.nn as nn
from main import attach_hook, current_analysis_context # Assuming main.py can be imported
from app.core.graph_parser import MockLensObject, MockLensNode # Reusing mocks

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 2)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class TestMainHooks(unittest.TestCase):
    def setUp(self):
        # Reset context before each test
        self.model = SimpleModel()
        # Simulate analysis
        # For attach_hook, 'lens_object' needs 'layer_nodes' with 'uid' and 'full_name'
        # and 'original_model' needs to be the actual model.
        layer_nodes_data = [
            {'uid': 'uid_lin1', 'name': 'linear1', 'node_type': 'Linear', 'full_name': 'linear1'},
            {'uid': 'uid_relu', 'name': 'relu', 'node_type': 'ReLU', 'full_name': 'relu'},
            {'uid': 'uid_lin2', 'name': 'linear2', 'node_type': 'Linear', 'full_name': 'linear2'},
        ]
        mock_lens = MockLensObject(layer_nodes_data=layer_nodes_data)

        current_analysis_context['original_model'] = self.model
        current_analysis_context['lens_object'] = mock_lens
        current_analysis_context['parsed_nodes'] = [] # Not strictly needed for these tests
        current_analysis_context['parsed_edges'] = []
        current_analysis_context['active_hooks'] = {}
        
        self.hook_called_count = 0

    def tearDown(self):
        # Clean up hooks from the model
        for handle_info in current_analysis_context['active_hooks'].values():
            if hasattr(handle_info, 'remove'): # Check if it's a hook handle
                handle_info.remove()
        current_analysis_context['active_hooks'].clear()


    def sample_hook_fn_str(self, an_id): # Renamed to avoid collision
        # Basic hook that increments a counter on self
        return f"""
def actual_hook_func(module, input, output):
    # print(f'Hook {an_id} called on {module}')
    # To access TestMainHooks.hook_called_count, it's tricky due to exec scope.
    # For testing, we'll check hook registration, not necessarily execution via this counter.
    # A global counter or a more complex setup would be needed if exec has to modify test case state.
    return output
"""

    def test_attach_forward_hook_success(self):
        hook_str = self.sample_hook_fn_str('test1')
        result = attach_hook('uid_lin1', hook_str, 'forward')
        self.assertTrue(result)
        self.assertIn('uid_lin1', current_analysis_context['active_hooks'])
        self.assertTrue(hasattr(self.model.linear1, '_forward_hooks'))
        self.assertEqual(len(self.model.linear1._forward_hooks), 1)

    def test_attach_backward_hook_success(self):
        hook_str = self.sample_hook_fn_str('test2') # Needs 3 args for backward
        # PyTorch backward hooks take (module, grad_input, grad_output)
        # grad_input can be a tuple, grad_output is a tuple.
        hook_str = """
def actual_hook_func(module, grad_input, grad_output):
    # print(f'Backward hook called on {module}')
    return None # Backward hooks can modify grad_input or return new grad_output
"""
        result = attach_hook('uid_lin2', hook_str, 'backward')
        self.assertTrue(result)
        self.assertIn('uid_lin2', current_analysis_context['active_hooks'])
        self.assertTrue(hasattr(self.model.linear2, '_full_backward_hooks')) # actually _full_backward_hooks
        self.assertEqual(len(self.model.linear2._full_backward_hooks), 1)


    def test_attach_hook_model_not_analyzed(self):
        current_analysis_context['original_model'] = None # Simulate not analyzed
        result = attach_hook('uid_lin1', self.sample_hook_fn_str('test3'), 'forward')
        self.assertFalse(result)

    def test_attach_hook_invalid_node_uid(self):
        result = attach_hook('invalid_uid', self.sample_hook_fn_str('test4'), 'forward')
        self.assertFalse(result)

    def test_attach_hook_bad_function_string(self):
        result = attach_hook('uid_lin1', "def badly_formatted_hook(module, input, output", 'forward')
        self.assertFalse(result)
        
    def test_attach_hook_replaces_existing(self):
        hook_str1 = self.sample_hook_fn_str('test_rep1')
        attach_hook('uid_relu', hook_str1, 'forward')
        self.assertEqual(len(self.model.relu._forward_hooks), 1)
        handle1_id = id(list(self.model.relu._forward_hooks.values())[0])


        hook_str2 = self.sample_hook_fn_str('test_rep2')
        attach_hook('uid_relu', hook_str2, 'forward') # Attach to same node
        self.assertEqual(len(self.model.relu._forward_hooks), 1) # Should still be 1
        handle2_id = id(list(self.model.relu._forward_hooks.values())[0])
        self.assertNotEqual(handle1_id, handle2_id, "New hook handle should be different, old one removed.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # For running in some environments
