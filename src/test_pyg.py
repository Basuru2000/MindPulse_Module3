import torch
from torch_geometric.nn import GCNConv

num_nodes = 46
edge_index = torch.randint(0, num_nodes, (2, 488))
x = torch.rand(num_nodes, 20)

conv1 = GCNConv(20, 64)
conv2 = GCNConv(64, 32)

h = torch.relu(conv1(x, edge_index))
out = conv2(h, edge_index)
print("Output shape:", out.shape)