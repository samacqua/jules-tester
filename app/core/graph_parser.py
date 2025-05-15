import torch
# We expect torchlens.Lens object, but to avoid circular dependencies or issues if torchlens
# is not installed when this module is imported, we won't type hint it directly here.
# from torchlens.lens import Lens # Avoid this for now.

def parse_torchlens_object(lens_object):
    """
    Parses the Lens object from torchlens to extract graph data (nodes and edges).

    Args:
        lens_object: The Lens object obtained from torchlens.log_model().

    Returns:
        A tuple (nodes, edges):
        - nodes: A list of dictionaries, where each dictionary represents a layer node.
                 Example: {'id': 'layer_uid_1', 'label': 'Conv2d_1', 'type': 'Conv2d', 
                            'full_name': 'module.conv1', 'details': {...}}
        - edges: A list of tuples, where each tuple represents a directed edge
                 (source_node_id, target_node_id).
    """
    nodes = []
    edges = []

    if not lens_object:
        return nodes, edges

    # --- Extract Layer Nodes ---
    # Assuming lens_object.layer_nodes is a list of objects/dictionaries
    # Each layer_node is assumed to have: uid, name, type (class name), and other details.
    # And crucially, information about its connections (e.g., parent_uids, child_uids, or via tensor_nodes)

    processed_layer_nodes = {} # To store extracted info, keyed by uid

    if not hasattr(lens_object, 'layer_nodes') or not hasattr(lens_object, 'tensor_nodes'):
        print("Warning: Lens object does not have 'layer_nodes' or 'tensor_nodes' attributes as expected.")
        # Attempt to use graph_summary if available, though it's less detailed
        if hasattr(lens_object, 'graph_summary') and isinstance(lens_object.graph_summary, dict):
             # This is a fallback and might not give a proper graph structure
            for i, (name, summary) in enumerate(lens_object.graph_summary.items()):
                nodes.append({
                    'id': name, # graph_summary keys might be unique enough
                    'label': name,
                    'type': summary.get('type', 'Unknown'),
                    'details': summary,
                    'full_name': name,
                })
            # No edge information from graph_summary directly in a simple format.
            return nodes, [] # Return early if only graph_summary is available
        return nodes, edges


    # First pass: gather all layer nodes
    for i, layer_node_obj in enumerate(lens_object.layer_nodes):
        # Assumptions about layer_node_obj attributes:
        uid = getattr(layer_node_obj, 'uid', f"layer_{i}") # Fallback uid
        name = getattr(layer_node_obj, 'name', f"Layer {i}")
        node_type = getattr(layer_node_obj, 'type', type(layer_node_obj).__name__)
        full_name = getattr(layer_node_obj, 'full_name', name) # e.g. "Sequential_0.Conv2d_0"

        # Store other details if available (e.g., parameters, output shape)
        details = {}
        for attr in ['params', 'output_shape', 'is_activation', 'depth']:
            if hasattr(layer_node_obj, attr):
                details[attr] = getattr(layer_node_obj, attr)
        
        # Add extra info from the 'raw_node' if it exists (torchlens specific)
        if hasattr(layer_node_obj, 'raw_node'):
             raw_node_info = getattr(layer_node_obj, 'raw_node', {})
             if isinstance(raw_node_info, dict): # Check if it's a dict as expected
                details['raw_node_classname'] = raw_node_info.get('classname')
                    details['raw_node_varname'] = raw_node_info.get('varname')
                # Ensure type is the classname if available and more specific
                if raw_node_info.get('classname'):
                    node_type = raw_node_info.get('classname')


        node_data = {
            'id': uid,
            'label': name,
            'type': node_type,
            'full_name': full_name,
            'details': details,
        }
        nodes.append(node_data)
        processed_layer_nodes[uid] = node_data

    # Second pass (if tensor_nodes exist): determine edges using tensor connections
    # Assumptions about tensor_node_obj attributes:
    #   uid, shape, dtype, producer_layer_uid (or similar), consumer_layer_uids (or similar)
    if hasattr(lens_object, 'tensor_nodes'):
        for tensor_node_obj in lens_object.tensor_nodes:
            producer_uid = getattr(tensor_node_obj, 'created_by_layer_uid', None) # Example attribute name
            if producer_uid is None: # Try another common name from torchlens
                producer_uid = getattr(tensor_node_obj, 'producer_uid', None)
            
            consumer_uids = getattr(tensor_node_obj, 'used_by_layer_uids', []) # Example attribute name
            if not consumer_uids: # Try another common name
                consumer_uids = getattr(tensor_node_obj, 'consumer_uids', [])

            if producer_uid and producer_uid in processed_layer_nodes:
                for consumer_uid in consumer_uids:
                    if consumer_uid in processed_layer_nodes:
                        # Check if edge already exists to avoid duplicates
                        edge = (producer_uid, consumer_uid)
                        if edge not in edges:
                            edges.append(edge)
    else:
        # Fallback: If no tensor_nodes, try to infer from parent/child relationships if they exist on layer_nodes
        # This is highly speculative as torchlens structure is not fully known yet
        for i, layer_node_obj in enumerate(lens_object.layer_nodes):
            current_uid = getattr(layer_node_obj, 'uid', f"layer_{i}")
            children_uids = getattr(layer_node_obj, 'children_uids', []) # Speculative
            parent_uids = getattr(layer_node_obj, 'parent_uids', []) # Speculative

            for child_uid in children_uids:
                if current_uid in processed_layer_nodes and child_uid in processed_layer_nodes:
                    edge = (current_uid, child_uid)
                    if edge not in edges:
                        edges.append(edge)
            
            for parent_uid in parent_uids:
                if parent_uid in processed_layer_nodes and current_uid in processed_layer_nodes:
                    edge = (parent_uid, current_uid)
                    if edge not in edges:
                        edges.append(edge)
                        
    # If no edges were found via tensor_nodes or parent/child attributes,
    # and the graph is sequential, try to link them in order.
    if not edges and len(nodes) > 1:
        print("Attempting to link nodes sequentially as a fallback.")
        for i in range(len(nodes) - 1):
            edges.append((nodes[i]['id'], nodes[i+1]['id']))


    return nodes, edges

if __name__ == '__main__':
    # This section is for basic testing of the parser if run directly.
    # It requires a mock Lens object.
    print("Graph Parser - Basic Test Mode")

    # Mocking a very simplified LensObject structure based on anticipated attributes
    class MockLensNode:
        def __init__(self, uid, name, node_type, children_uids=None, parent_uids=None, raw_node=None, params=0, output_shape=None):
            self.uid = uid
            self.name = name
            self.type = node_type
            self.full_name = name
            self.children_uids = children_uids if children_uids else []
            self.parent_uids = parent_uids if parent_uids else []
            self.raw_node = raw_node if raw_node else {} # Example: {'classname': 'Linear', 'varname': 'fc1'}
            self.params = params
            self.output_shape = output_shape

    class MockTensorNode:
        def __init__(self, uid, producer_uid, consumer_uids):
            self.uid = uid
            self.producer_uid = producer_uid # Using 'producer_uid' as one guess
            self.consumer_uids = consumer_uids # Using 'consumer_uids' as one guess


    class MockLensObject:
        def __init__(self, layer_nodes_data, tensor_nodes_data=None):
            self.layer_nodes = [MockLensNode(**data) for data in layer_nodes_data]
            if tensor_nodes_data:
                self.tensor_nodes = [MockTensorNode(**data) for data in tensor_nodes_data]
            else:
                self.tensor_nodes = []
        
        # Add graph_summary for fallback testing
        @property
        def graph_summary(self):
            summary = {}
            for node in self.layer_nodes:
                summary[node.name] = {'type': node.type, 'params': node.params, 'output_shape': node.output_shape}
            return summary


    # Test case 1: Simple sequential model using layer_nodes parent/child (less likely for torchlens)
    mock_layers_1 = [
        {'uid': 'l1', 'name': 'InputLayer', 'node_type': 'Input', 'children_uids': ['l2']},
        {'uid': 'l2', 'name': 'ConvLayer', 'node_type': 'Conv2d', 'parent_uids': ['l1'], 'children_uids': ['l3'], 'raw_node': {'classname': 'Conv2d'}},
        {'uid': 'l3', 'name': 'OutputLayer', 'node_type': 'Linear', 'parent_uids': ['l2'], 'raw_node': {'classname': 'Linear'}},
    ]
    mock_lens_1 = MockLensObject(mock_layers_1)
    
    # Test case 2: Model with tensor connections (more likely for torchlens)
    mock_layers_2 = [
        {'uid': 'fc1', 'name': 'FullyConnected1', 'node_type': 'Linear', 'raw_node': {'classname': 'Linear', 'varname': 'fc1'}},
        {'uid': 'relu1', 'name': 'ReLU1', 'node_type': 'ReLU', 'raw_node': {'classname': 'ReLU', 'varname': 'relu1'}},
        {'uid': 'fc2', 'name': 'FullyConnected2', 'node_type': 'Linear', 'raw_node': {'classname': 'Linear', 'varname': 'fc2'}},
    ]
    mock_tensors_2 = [
        {'uid': 't1', 'producer_uid': 'fc1', 'consumer_uids': ['relu1']},
        {'uid': 't2', 'producer_uid': 'relu1', 'consumer_uids': ['fc2']},
    ]
    mock_lens_2 = MockLensObject(mock_layers_2, mock_tensors_2)

    print("\n--- Testing with Mock Lens Object 1 (Parent/Child links) ---")
    nodes1, edges1 = parse_torchlens_object(mock_lens_1)
    print("Nodes:", nodes1)
    print("Edges:", edges1)

    print("\n--- Testing with Mock Lens Object 2 (Tensor connections) ---")
    nodes2, edges2 = parse_torchlens_object(mock_lens_2)
    print("Nodes:", nodes2)
    print("Edges:", edges2)
    
    print("\n--- Testing with None object ---")
    nodes_none, edges_none = parse_torchlens_object(None)
    print("Nodes:", nodes_none)
    print("Edges:", edges_none)

    print("\n--- Testing with object missing tensor_nodes (Fallback to sequential if no parent/child) ---")
    mock_lens_3 = MockLensObject(mock_layers_1) # No tensor data, no parent/child on nodes themselves
    # To simulate no parent/child on nodes for this test, let's clear them
    for node_obj in mock_lens_3.layer_nodes:
        node_obj.parent_uids = []
        node_obj.children_uids = []
    nodes3, edges3 = parse_torchlens_object(mock_lens_3)
    print("Nodes:", nodes3)
    print("Edges:", edges3)
