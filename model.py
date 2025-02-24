import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaCGN(nn.Module):
    def __init__(self, input_dim, node_dim, n_layers) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.n_layers = n_layers
        self.U0 = nn.init.kaiming_normal_(torch.empty(node_dim, input_dim))
        self.b0 = nn.Parameter(torch.randn(node_dim))
        self.convLayers = nn.ModuleList(
            [ConvNetLayer(self.node_dim) for _ in range(self.n_layers)]
        )

    def forward(self, x, adj_mat):
        x = x @ self.U0 + self.b0  # self.b0 is broadcasted properly?
        for i in range(self.n_layers):
            x = self.convLayers[i](x, adj_mat)
        return x


class ConvNetLayer(nn.Module):
    def __init__(self, node_dim) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.randn(node_dim, node_dim))

    def forward(self, x, adj_mat):
        new_x = torch.empty_like(x)
        for i in range(x.shape[0]):
            deg_i = adj_mat[:, i].sum()
            mask_i = adj_mat[:, i] > 0
            new_x[i, :] = F.relu(
                self.U @ (x[mask_i, :].sum(dim=0)).to(torch.float32) / deg_i
            )
        return new_x


if __name__ == "__main__":
    adj_mat = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    X = torch.tensor([[5, 2], [3, 4], [10, 20]], dtype=torch.float32)
    layer = ConvNetLayer(node_dim=2)

    X = layer(X)
    print(X)

    model = VanillaCGN(input_dim=2, node_dim=2, n_layers=2)
    Y = torch.rand((3, 2))
    Y = model(Y)
    print(Y)
