import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaCGN(nn.Module):
    def __init__(self, input_dim, node_dim, n_layers) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.n_layers = n_layers
        self.U0 = nn.init.xavier_uniform_(torch.empty(input_dim, node_dim))
        self.b0 = nn.Parameter(nn.init.uniform_(torch.empty(node_dim)))
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
        new_x = torch.empty_like(x)
        node_degree = adj_mat.sum(axis=1)
        for i in range(new_x.shape[1]):
            deg_i = node_degree[:, i]
            mask_i = adj_mat[:, :, i] > 0
            mask_i = mask_i.unsqueeze(2).expand(-1, -1, x.size(2))
            new_x[:, i, :] = F.relu(
                ((x * mask_i).sum(axis=1)) @ self.U / deg_i.unsqueeze(1)
            ).squeeze()
        return new_x


class GraphRegressionReadoutLayer(nn.Module):
    def __init__(self, node_dim) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.Q = nn.Parameter(nn.init.xavier_normal_(torch.empty(node_dim, node_dim)))
        self.P = nn.Parameter(nn.init.xavier_normal_(torch.empty((1, node_dim))))

    def forward(self, x):
        x = x.sum(dim=1) / x.shape[1]  # dim 0 is batch size so we collapse along dim 1
        return (self.P @ F.relu(self.Q @ x.T)).squeeze()
