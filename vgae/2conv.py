import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionLayer, self).__init__()
        layers = []
        layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        in_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EdgeConvolutionModel(nn.Module):
    def __init__(self, input_size):
        super(EdgeConvolutionModel, self).__init__()
        self.conv1 = ConvolutionLayer(1, 64)
        self.conv2 = ConvolutionLayer(64, 10)

    def forward(self, x):
        x = x.view(1, 1, x.size(0), x.size(1))  # Add batch and channel dimensions
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(10, x.size(2), x.size(3))

# Example usage
input_matrix = torch.rand(2708, 2708)  # Assuming values in the range [0, 1]

model = EdgeConvolutionModel(input_size=input_matrix.size())
print(EdgeConvolutionModel)

output_tensor = model(input_matrix)
print(output_tensor.size())
