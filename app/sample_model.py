import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10, 20) # Input size 10, output size 20
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)  # Input size 20, output size 5

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

if __name__ == '__main__':
    # Example usage (optional, for direct testing of this file)
    model = SimpleNN()
    print("SimpleNN model defined and instantiated:")
    print(model)
    # Dummy input
    dummy_input = torch.randn(1, 10) # Batch size 1, input features 10
    output = model(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
