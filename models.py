import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(output_dim, input_dim))
        )
        self.a1 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(output_dim, 1))
        )  # a = concat(a1, a2)
        self.a2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(output_dim, 1)))

    def forward(self, x, adj_mat):
        x = x @ self.W.T  # matrix containing the W*hi vectors
        f1 = (x @ self.a1).squeeze()
        f2 = (x @ self.a2).squeeze()
        e = F.leaky_relu_(f1.unsqueeze(1) + f2.unsqueeze(0))  # Broadcast
        e = e * adj_mat  # ensures that the softmax is only performed over the neighbors
        e = torch.where(e == 0, -torch.inf, e)
        alpha = F.softmax(e, dim=1)  # or dim=0?
        return alpha @ x


# TODO: Extend to more than 2 layers
class GAT(nn.Module):
    def __init__(self, dimensions: list[list[int, int]], n_heads: list[int]):
        super(GAT, self).__init__()
        self.attention1 = nn.ModuleList(
            AttentionLayer(dimensions[0][0], dimensions[0][1])
            for _ in range(n_heads[0])
        )
        # only one head for second layer
        self.attention2 = nn.ModuleList(
            AttentionLayer(dimensions[1][0], dimensions[1][1])
            for _ in range(n_heads[1])
        )

    def forward(self, data):
        x, adj_mat = data.x, data.adj_mat
        x = torch.cat([attention(x, adj_mat) for attention in self.attention1], dim=1)
        x = F.elu(x)
        x = torch.cat([attention(x, adj_mat) for attention in self.attention2], dim=1)
        x = F.elu(x)
        return x


class EdgePrediction(nn.Module):
    """Takes x = concat[hi, hj] node embeddings as inputs and returns a score predicting the likelihood of an edge between i and j
    (cf. p13 benchmarking GNNs for notations)
    P output dim is 1 because this is a binary classification task
    Q output dim is embedding_dim but could IN THEORY BE ANYTHING

    input_dim: int, dimension of the input node embeddings"""

    def __init__(self, embedding_dim):
        super(EdgePrediction, self).__init__()
        self.Q = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty((embedding_dim * 2, embedding_dim)))
        )
        self.P = nn.Parameter(nn.init.xavier_uniform_(torch.empty((embedding_dim, 1))))

    def forward(self, x):
        x = F.relu(x @ self.Q)
        x = x @ self.P
        return x.squeeze()


class GCNLayer(nn.Module):
    """For now, set the hidden dim to be the avg of input and output dim, to smooth the dimension variation"""

    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.W0 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(input_dim, (output_dim + input_dim) // 2)
            )
        )
        self.W1 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((output_dim + input_dim) // 2, output_dim)
            )
        )

    def forward(self, x, A_hat):
        x = F.relu(A_hat @ x @ self.W0)
        return A_hat @ x @ self.W1


class GCN(nn.Module):
    def __init__(self, dimensions: list[tuple[int, int]], dropout):
        """dimensions: list of tuples (input_dim, output_dim) for each layer"""
        super(GCN, self).__init__()
        self.layers = nn.ModuleList(
            [
                GCNLayer(dimensions[i][0], dimensions[i][1])
                for i in range(len(dimensions))
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, A_hat = data.x, data.A_hat
        for i, layer in enumerate(self.layers):
            x = layer(x, A_hat)
            if i != len(self.layers) - 1:
                x = self.dropout(x)
        return x
