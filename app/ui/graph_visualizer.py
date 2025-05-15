import tkinter as tk
from tkinter import ttk
import networkx as nx
import pprint
from tkinter import simpledialog, messagebox

class GraphVisualizerApp:
    def __init__(self, root, graph_data=None, hook_attachment_callback=None):
        self.root = root
        self.root.title("PyTorch Model Graph Visualizer")
        self.root.geometry("800x600")

        self.graph_data = graph_data if graph_data else {'nodes': [], 'edges': []}
        self.hook_attachment_callback = hook_attachment_callback
        self.selected_node_id = None
        self.hooked_node_uids = set() # For tracking hooked nodes

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Canvas for graph drawing
        self.canvas = tk.Canvas(main_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))

        # Sidebar for node inspection
        self.sidebar = ttk.Frame(main_frame, width=250, padding="10")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        
        sidebar_label = ttk.Label(self.sidebar, text="Node Inspector", font=("Arial", 14))
        sidebar_label.pack(pady=10)
        
        self.node_info_text = tk.Text(self.sidebar, height=10, width=30, state=tk.DISABLED)
        self.node_info_text.pack(pady=10)

        self.attach_hook_button = ttk.Button(self.sidebar, text="Attach Hook to Selected Node", command=self.on_attach_hook_button_click, state=tk.DISABLED)
        self.attach_hook_button.pack(pady=10)

        # Placeholder for graph drawing
        self.draw_graph()

    def set_graph_data(self, nodes, edges):
        self.graph_data['nodes'] = nodes
        self.graph_data['edges'] = edges
        self.draw_graph() # Redraw graph with new data

    def draw_graph(self):
        self.canvas.delete("all") # Clear previous drawings

        if not self.graph_data['nodes']:
            self.canvas.create_text(400, 300, text="No graph data to display.", font=("Arial", 16))
            return

        node_positions = {}
        layout_successful = False
        # Define node_width and node_height here as they are used in fallback and drawing
        node_width = 120
        node_height = 50

        try:
            # import networkx as nx # Already imported at the top
            
            G = nx.DiGraph()
            for node_data in self.graph_data['nodes']:
                G.add_node(node_data['id']) 
            
            for edge_data in self.graph_data['edges']:
                G.add_edge(edge_data[0], edge_data[1])

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1: canvas_width = 700 
            if canvas_height <= 1: canvas_height = 500 

            try:
                pos = nx.kamada_kawai_layout(G) 
            except Exception as e_layout: 
                print(f"Kamada-Kawai layout failed: {e_layout}, trying spring_layout.")
                pos = nx.spring_layout(G, k=0.15, iterations=20)

            # Scale positions to fit canvas (with some padding)
            # Check if pos is not empty before trying to access values
            if not pos: # Handles single-node graphs or other empty pos scenarios
                if G.number_of_nodes() == 1: # Single node special case
                    # Place single node in the center
                    node_positions = {list(G.nodes())[0]: (canvas_width / 2, canvas_height / 2)}
                    layout_successful = True
                else: # No nodes or other issue
                    raise ValueError("Layout position calculation failed or graph is empty.")

            else: # Original scaling logic for multiple nodes
                min_x = min(p[0] for p in pos.values())
                max_x = max(p[0] for p in pos.values())
                min_y = min(p[1] for p in pos.values())
                max_y = max(p[1] for p in pos.values())
                
                range_x = max_x - min_x
                range_y = max_y - min_y
                
                if range_x == 0: range_x = 1 
                if range_y == 0: range_y = 1

                padding = 50 
                
                def scale_pos(p_val, p_min, p_range, canvas_dim_size):
                    return padding + ((p_val - p_min) / p_range) * (canvas_dim_size - 2 * padding)

                node_positions = {
                    node_id: (
                        scale_pos(p[0], min_x, range_x, canvas_width),
                        scale_pos(p[1], min_y, range_y, canvas_height)
                    ) for node_id, p in pos.items()
                }
                layout_successful = True

        except ImportError:
            print("NetworkX library not found. Falling back to simple list layout.")
            node_positions = {} 
            layout_successful = False
        except Exception as e: 
            print(f"Error during NetworkX layout: {e}. Falling back to simple list layout.")
            node_positions = {} 
            layout_successful = False

        if not layout_successful:
            current_y = 50
            x_center = (self.canvas.winfo_width() / 2) if self.canvas.winfo_width() > 1 else 400
            for i, node in enumerate(self.graph_data['nodes']):
                node_id = node['id']
                node_positions[node_id] = (x_center, current_y + node_height / 2) 
                current_y += node_height + 30 
        
        drawn_nodes = {} 
        for node_data in self.graph_data['nodes']:
            node_id = node_data['id']
            label = node_data.get('label', node_id)
            node_type = node_data.get('type', 'Unknown')
            
            center_x, center_y = node_positions[node_id]
            
            x0 = center_x - node_width / 2
            y0 = center_y - node_height / 2
            x1 = center_x + node_width / 2
            y1 = center_y + node_height
            
            node_type_short = node_type
            if len(node_type_short) > 15:
                 node_type_short = node_type[:12] + "..."
            
            node_fill_color = "lightgreen" if node_id in self.hooked_node_uids else "skyblue"

            rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, fill=node_fill_color, outline="black", tags=(node_id, "graph_node"))
            text_id = self.canvas.create_text(center_x, center_y, text=f"{label}\n({node_type_short})", tags=(node_id, "graph_node_label"), justify=tk.CENTER)
            
            drawn_nodes[node_id] = {'rect': rect_id, 'text': text_id}
            
            self.canvas.tag_bind(rect_id, "<Button-1>", lambda event, nid=node_id: self.on_node_click(nid))
            self.canvas.tag_bind(text_id, "<Button-1>", lambda event, nid=node_id: self.on_node_click(nid))

        for edge in self.graph_data['edges']:
            source_id, target_id = edge
            if source_id in node_positions and target_id in node_positions:
                x_start, y_start = node_positions[source_id]
                x_end, y_end = node_positions[target_id]
                
                dx = x_end - x_start
                dy = y_end - y_start
                length = (dx**2 + dy**2)**0.5
                
                x_start_adj, y_start_adj = x_start, y_start
                x_end_adj, y_end_adj = x_end, y_end

                if length > 0: 
                    if length > node_width/2 : 
                         x_start_adj = x_start + (dx * node_width/2) / length
                         y_start_adj = y_start + (dy * node_height/2) / length
                    # No else needed, default is x_start, y_start

                    if length > node_width/2: 
                         x_end_adj = x_end - (dx * node_width/2) / length
                         y_end_adj = y_end - (dy * node_height/2) / length
                    # No else needed, default is x_end, y_end
                
                self.canvas.create_line(x_start_adj, y_start_adj, x_end_adj, y_end_adj, arrow=tk.LAST, fill="slateGray", width=1.5)
        
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def on_node_click(self, node_id):
        node_data = next((n for n in self.graph_data['nodes'] if n['id'] == node_id), None)
        
        self.node_info_text.config(state=tk.NORMAL)
        self.node_info_text.delete(1.0, tk.END)
        if node_data:
            info = f"ID: {node_data['id']}\n"
            info += f"Label: {node_data.get('label', 'N/A')}\n"
            info += f"Type: {node_data.get('type', 'N/A')}\n"
            info += f"Full Name: {node_data.get('full_name', 'N/A')}\n"
            info += "\nDetails:\n"
            details_dict = node_data.get('details', {})
            if not details_dict:
                info += "  (No additional details available)\n"
            else:
                for key, value in details_dict.items():
                    if isinstance(value, (dict, list)):
                        formatted_value = pprint.pformat(value, indent=2) # Indent of 2 for the value itself
                        info += f"  {key}:\n    {formatted_value.replace('\n', '\n    ')}\n" # Indent formatted value under key
                    else:
                        info += f"  {key}: {value}\n"
            
            if node_id in self.hooked_node_uids:
                info += "\nStatus: Hook currently attached\n"
            
            self.node_info_text.insert(tk.END, info)
        else:
            self.node_info_text.insert(tk.END, "Node data not found.")
        self.node_info_text.config(state=tk.DISABLED)
        print(f"Node clicked: {node_id}") # For console debugging
        self.selected_node_id = node_id
        if self.attach_hook_button:
            self.attach_hook_button.config(state=tk.NORMAL)

    def on_attach_hook_button_click(self):
        if self.selected_node_id is None:
            messagebox.showerror("Error", "No node selected. Please click on a node first.")
            return
        
        if self.hook_attachment_callback is None:
            messagebox.showerror("Error", "Hook attachment callback not configured.")
            return
        
        self.prompt_for_hook_details(self.selected_node_id)

    def prompt_for_hook_details(self, node_id):
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Attach Hook to {node_id}")
        dialog.geometry("500x450") # Adjusted size
        dialog.transient(self.root) # Keep dialog on top of the main window
        dialog.grab_set() # Modal behavior

        ttk.Label(dialog, text=f"Attaching hook to Node ID: {node_id}").pack(pady=5)

        # Hook Type
        ttk.Label(dialog, text="Hook Type:").pack(pady=(10,0))
        hook_type_var = tk.StringVar(value='forward')
        hook_type_combo = ttk.Combobox(dialog, textvariable=hook_type_var, values=['forward', 'backward'], state='readonly')
        hook_type_combo.pack()

        # Hook Function String
        ttk.Label(dialog, text="Hook Function (e.g., def my_hook(module, input, output): ...):").pack(pady=(10,0))
        
        # Frame for Text widget and Scrollbar
        text_frame = ttk.Frame(dialog)
        text_frame.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)

        hook_function_text = tk.Text(text_frame, height=15, width=60, wrap=tk.WORD) # Increased height
        scrollbar = ttk.Scrollbar(text_frame, command=hook_function_text.yview)
        hook_function_text['yscrollcommand'] = scrollbar.set
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        hook_function_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        hook_function_text.insert(tk.END, "def new_hook(module, input, output):\n    # 'input' is a tuple of tensors\n    # 'output' is a tensor (for forward) or tuple of tensors (for backward)\n    print(f'Hook triggered for {module._get_name()} on node {node_id}')\n    print(f'  Input shapes: {[i.shape for i in input if hasattr(i, "shape")]}')\n    print(f'  Output shape/grad: {output.shape if hasattr(output, "shape") else [o.shape for o in output if hasattr(o, "shape")]}')\n    return output")


        def on_apply():
            hook_type = hook_type_var.get()
            hook_function_str = hook_function_text.get(1.0, tk.END).strip()

            if not hook_function_str:
                messagebox.showerror("Error", "Hook function string cannot be empty.", parent=dialog)
                return

            if self.hook_attachment_callback:
                try:
                    success = self.hook_attachment_callback(node_id, hook_function_str, hook_type)
                    if success:
                        self.hooked_node_uids.add(node_id)
                        messagebox.showinfo("Hook Status", f"Successfully attached {hook_type} hook to {node_id}.", parent=dialog)
                        self.draw_graph() # Refresh graph
                        dialog.destroy()
                    else:
                        # attach_hook in main.py should already print detailed errors to console
                        messagebox.showerror("Hook Status", f"Failed to attach {hook_type} hook to {node_id}. Check console for details.", parent=dialog)
                        # self.hooked_node_uids.discard(node_id) # Optional: if a re-hook attempt fails
                        # self.draw_graph() 
                except Exception as e: # Should ideally be caught by attach_hook and return False
                    messagebox.showerror("Error", f"An unexpected error occurred calling the hook callback: {e}", parent=dialog)
            else:
                messagebox.showerror("Error", "Hook callback not configured.", parent=dialog)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Apply Hook", command=on_apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

def launch_ui(graph_nodes=None, graph_edges=None, hook_callback=None):
    """Launches the Tkinter UI with optional graph data."""
    root = tk.Tk()
    initial_data = {'nodes': graph_nodes if graph_nodes else [], 'edges': graph_edges if graph_edges else []}
    app = GraphVisualizerApp(root, graph_data=initial_data, hook_attachment_callback=hook_callback)
    root.mainloop()

if __name__ == '__main__':
    # Example Usage (for testing app/ui/graph_visualizer.py directly)
    print("Launching Graph Visualizer with sample data...")
    sample_nodes = [
        {'id': 'layer1', 'label': 'Conv2D', 'type': 'Convolutional', 'full_name': 'model.conv1', 'details': {'filters': 32, 'kernel': (3,3)}},
        {'id': 'layer2', 'label': 'ReLU', 'type': 'Activation', 'full_name': 'model.relu1', 'details': {}},
        {'id': 'layer3', 'label': 'Linear', 'type': 'Dense', 'full_name': 'model.fc1', 'details': {'units': 10}},
    ]
    sample_edges = [('layer1', 'layer2'), ('layer2', 'layer3')]
    launch_ui(sample_nodes, sample_edges)
