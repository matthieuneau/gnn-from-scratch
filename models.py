import torch
import torch.nn as nn
import torch.nn.functional as F

from GAT import AttentionLayer


# TODO: Extend to more than 2 layers
class GAT(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_heads_1, n_heads_2):
        super(GAT, self).__init__()
        self.attention1 = nn.ModuleList(
            AttentionLayer(node_dim, hidden_dim) for _ in range(n_heads_1)
        )
        # only one head for second layer
        self.attention2 = nn.ModuleList(
            AttentionLayer(hidden_dim * n_heads_1, hidden_dim) for _ in range(n_heads_2)
        )

    def forward(self, data):
        x, adj_mat = data.x, data.adj_mat
        residual = x
        x = torch.cat([attention(x, adj_mat) for attention in self.attention1], dim=1)
        x = F.elu(x) + residual
        residual = x
        x = torch.cat([attention(x, adj_mat) for attention in self.attention2], dim=1)
        x = F.elu(x) + residual
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

    def forward(self, data):
        x, A_hat = data.x, data.A_hat
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
        for layer in self.layers:
            x = layer(x, A_hat)
            x = self.dropout(x)
        return x
