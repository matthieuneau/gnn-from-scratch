import torch
import torch.nn as nn

import torch.nn.functional as F


class VanillaCGN(nn.Module):
    def __init__(self, input_dim, node_dim, n_layers, adj_mat) -> None:
        super().__init__()
        self.adj_mat = adj_mat
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.n_layers = n_layers
        self.U0 = nn.init.kaiming_normal(self.node_dim, self.input_dim)
        self.b0 = nn.init.kaiming_normal(self.node_dim)
        self.convLayers = [
            ConvNetLayer(self.node_dim, self.adj_mat) for _ in range(self.n_layers)
        ]

    def forward(self, x):
        x = self.U0 @ x + self.b0
        for i in range(self.n_layers):
            x = self.convLayers[i](x)
        return x


class ConvNetLayer(nn.Module):
    def __init__(self, node_dim, adj_mat) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.randn(node_dim, node_dim))
        self.adj_mat = adj_mat

    def forward(self, x):
        old_x = x.clone()
        for i in range(x.shape[0]):
            deg_i = self.adj_mat[:, i].sum()  # we look at the neighbors pointing to i
            mask_i = self.adj_mat[:, i] > 0
            x[i, :] = F.relu(self.U @ (old_x[mask_i, :].sum(dim=0)).mT / deg_i)
        return x


adj_mat = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
X = torch.tensor([[1, 2], [3, 4], [5, 6]])
layer = ConvNetLayer(node_dim=2, adj_mat=adj_mat)

X = layer(X)
print(X)
