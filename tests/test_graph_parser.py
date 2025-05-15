import unittest
from app.core.graph_parser import parse_torchlens_object

# Mocking classes as defined in graph_parser.py's __main__ for testing
class MockLensNode:
    def __init__(self, uid, name, node_type, full_name=None, children_uids=None, parent_uids=None, raw_node=None, params=0, output_shape=None, **kwargs):
        self.uid = uid
        self.name = name
        self.type = node_type
        self.full_name = full_name if full_name else name
        self.children_uids = children_uids if children_uids else []
        self.parent_uids = parent_uids if parent_uids else []
        self.raw_node = raw_node if raw_node else {}
        self.params = params
        self.output_shape = output_shape
        # Allow arbitrary other attributes that might be on a real lens node
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockTensorNode:
    def __init__(self, uid, producer_uid, consumer_uids, **kwargs):
        self.uid = uid
        # Try various common names for producer/consumer from torchlens history/guesses
        self.created_by_layer_uid = producer_uid 
        self.used_by_layer_uids = consumer_uids
        self.producer_uid = producer_uid 
        self.consumer_uids = consumer_uids
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockLensObject:
    def __init__(self, layer_nodes_data=None, tensor_nodes_data=None, graph_summary_data=None):
        self.layer_nodes = [MockLensNode(**data) for data in layer_nodes_data] if layer_nodes_data else []
        self.tensor_nodes = [MockTensorNode(**data) for data in tensor_nodes_data] if tensor_nodes_data else []
        self.graph_summary = graph_summary_data if graph_summary_data else {}
        if not graph_summary_data and layer_nodes_data : # Auto-populate graph_summary if not given
             self.graph_summary = {node.name: {'type': node.type, 'params': node.params} for node in self.layer_nodes}


class TestGraphParser(unittest.TestCase):

    def test_parse_empty_lens_object(self):
        mock_lens = MockLensObject()
        nodes, edges = parse_torchlens_object(mock_lens)
        self.assertEqual(nodes, [])
        self.assertEqual(edges, [])

    def test_parse_lens_object_none(self):
        nodes, edges = parse_torchlens_object(None)
        self.assertEqual(nodes, [])
        self.assertEqual(edges, [])

    def test_parse_with_layer_nodes_only_sequential_fallback(self):
        layer_data = [
            {'uid': 'l1', 'name': 'Linear1', 'node_type': 'Linear', 'raw_node': {'classname': 'Linear'}},
            {'uid': 'l2', 'name': 'ReLU1', 'node_type': 'ReLU', 'raw_node': {'classname': 'ReLU'}},
            {'uid': 'l3', 'name': 'Linear2', 'node_type': 'Linear', 'raw_node': {'classname': 'Linear'}},
        ]
        mock_lens = MockLensObject(layer_nodes_data=layer_data)
        nodes, edges = parse_torchlens_object(mock_lens)
        
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0]['id'], 'l1')
        self.assertEqual(nodes[1]['type'], 'ReLU') # Checking raw_node.classname override
        
        # Fallback sequential edges
        self.assertEqual(edges, [('l1', 'l2'), ('l2', 'l3')])

    def test_parse_with_tensor_connections(self):
        layer_data = [
            {'uid': 'fc1', 'name': 'FC1', 'node_type': 'Linear', 'full_name': 'fc1_full'},
            {'uid': 'relu', 'name': 'ReLU1', 'node_type': 'ReLU', 'full_name': 'relu_full'},
            {'uid': 'fc2', 'name': 'FC2', 'node_type': 'Linear', 'full_name': 'fc2_full'},
        ]
        tensor_data = [
            {'uid': 't1', 'producer_uid': 'fc1', 'consumer_uids': ['relu']},
            {'uid': 't2', 'producer_uid': 'relu', 'consumer_uids': ['fc2']},
        ]
        mock_lens = MockLensObject(layer_nodes_data=layer_data, tensor_nodes_data=tensor_data)
        nodes, edges = parse_torchlens_object(mock_lens)

        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0]['full_name'], 'fc1_full')
        self.assertEqual(edges, [('fc1', 'relu'), ('relu', 'fc2')])
        
    def test_parse_with_graph_summary_fallback(self):
        summary_data = {
            "conv1": {"type": "Conv2d", "params": 100},
            "relu1": {"type": "ReLU"},
        }
        # Pass no layer_nodes or tensor_nodes, only graph_summary
        mock_lens = MockLensObject(graph_summary_data=summary_data)
        # Temporarily remove layer_nodes and tensor_nodes attributes for this specific test
        # as parse_torchlens_object checks for their presence first.
        del mock_lens.layer_nodes 
        del mock_lens.tensor_nodes

        nodes, edges = parse_torchlens_object(mock_lens)
        self.assertEqual(len(nodes), 2)
        self.assertTrue(any(n['id'] == 'conv1' and n['type'] == 'Conv2d' for n in nodes))
        self.assertEqual(edges, []) # No edges from graph_summary

if __name__ == '__main__':
    unittest.main()
