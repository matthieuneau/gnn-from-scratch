import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """For now, set the hidden dim to be the same as the output dim"""

    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.W0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, output_dim))
        )
        self.W1 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(output_dim, output_dim))
        )

    def forward(self, x, A_hat):
        x = F.relu(A_hat @ x @ self.W0)
        return A_hat @ x @ self.W1


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList(
            [
                GCNLayer(input_dim, hidden_dim),
                *[GCNLayer(hidden_dim, hidden_dim) for _ in range(n_layers - 2)],
                GCNLayer(hidden_dim, output_dim),
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_hat):
        for layer in self.layers:
            x = layer(x, A_hat)
            x = self.dropout(x)
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
