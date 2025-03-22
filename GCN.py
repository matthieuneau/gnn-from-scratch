import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_classes, dropout):
        super(GCN, self).__init__()
        self.W0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(node_dim, hidden_dim))
        )
        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(hidden_dim, n_classes))
        )

    def forward(self, x, A_hat):
        x = self.dropout(x)
        x = F.relu(A_hat @ x @ self.W0)
        x = self.dropout(x)
        logits = A_hat @ x @ self.W1
        return logits
