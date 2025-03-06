import torch
from torch.fx import node
import torch.nn as nn
import torch.nn.functional as F


class VanillaCGN(nn.Module):
    def __init__(self, input_dim, node_dim, n_layers) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.n_layers = n_layers
        self.U0 = nn.init.xavier_uniform_(torch.empty(input_dim, node_dim))
        self.b0 = nn.Parameter(torch.randn(node_dim) / 1e3)
        self.convLayers = nn.ModuleList(
            [ConvNetLayer(self.node_dim) for _ in range(self.n_layers)]
        )
        self.readOutLayer = GraphRegressionReadoutLayer(node_dim=node_dim)

    def forward(self, x, adj_mat):
        x = x @ self.U0 + self.b0  # self.b0 is broadcasted properly?
        for i in range(self.n_layers):
            x = self.convLayers[i](x, adj_mat)
        x = self.readOutLayer(x)
        return x


class ConvNetLayer(nn.Module):
    def __init__(self, node_dim) -> None:
        super().__init__()
        self.U = nn.Parameter(nn.init.xavier_normal_(torch.empty(node_dim, node_dim)))

    def forward(self, x, adj_mat):
        new_x = torch.empty_like(x.squeeze())
        for i in range(new_x.shape[0]):
            deg_i = adj_mat[:, i].sum()
            mask_i = adj_mat[:, i] > 0
            new_x[i, :] = F.leaky_relu_(self.U @ (x[mask_i, :].sum(dim=0)) / deg_i)
        new_x = new_x.unsqueeze(0)  # To return a 3d batch
        return new_x


class GraphRegressionReadoutLayer(nn.Module):
    def __init__(self, node_dim) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.Q = nn.Parameter(nn.init.xavier_normal_(torch.empty(node_dim, node_dim)))
        self.P = nn.Parameter(nn.init.xavier_normal_(torch.empty((1, node_dim))))

    def forward(self, x):
        x = x.sum(dim=1) / x.shape[1]  # dim 0 is batch size so we collapse along dim 1
        return (self.P @ F.leaky_relu_(self.Q @ x.T)).squeeze()


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.init.xavier_uniform_(torch.empty(output_dim, input_dim))
        self.a = nn.init.xavier_uniform_(torch.empty(@*output_dim, 1)))
        
    def forward(self, x):
        x = x @ self.W.T         # matrix containing the W*hi vectors
        


class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.fc3 = nn.Linear(50, 121)   # 121 classes for each node, 50 input dim

    def forward(self, x, edge_index):
        x = self.fc3(x)
        return x
