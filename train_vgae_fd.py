import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import VGAE
from torch_geometric.datasets import Planetoid
#from torch_geometric.data import train_test_split

# Load Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Split the dataset into training and test sets
# Define VGAE model
class MyEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyEncoder, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.conv_mu = nn.Linear(out_channels, out_channels)
        self.conv_logvar = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv_mu(x), self.conv_logvar(x)

# Instantiate VGAE with your encoder
my_encoder = MyEncoder(in_channels=dataset.num_features, out_channels=16)
vgae_model = VGAE(encoder=my_encoder)

# Define optimizer
optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.01)

# Training loop
def train():
    vgae_model.train()
    optimizer.zero_grad()
    z = vgae_model.encode(data.x, data.edge_index)
    loss = vgae_model.recon_loss(z, data.edge_index) + vgae_model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing loop
def test(mask):
    vgae_model.eval()
    z = vgae_model.encode(data.x, data.edge_index)
    logits = vgae_model.decode(z, data.edge_index)
    pred = logits[mask].argmax(dim=1)
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc

# Train and test loop
for epoch in range(1, 201):
    loss = train()
    train_acc = test(data.train_mask)
    val_acc = test(data.val_mask)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# Test the final model on the test set
test_acc = test(data.test_mask)
print(f'Final Test Accuracy: {test_acc:.4f}')
