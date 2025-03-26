import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AttentionLayer


class GATTransductive(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_classes, n_heads, dropout):
        super(GATTransductive, self).__init__()
        self.n_heads = n_heads
        self.attention1 = nn.ModuleList(
            AttentionLayer(node_dim, hidden_dim) for _ in range(n_heads)
        )
        self.dropout = nn.Dropout(dropout)
        # only one head for second layer
        self.attention2 = AttentionLayer(hidden_dim * n_heads, n_classes)
        # self.fc1 = nn.Linear(hidden_dim * n_heads, n_classes)

    def forward(self, x, adj_mat):
        x = torch.cat(
            [self.attention1[i](x, adj_mat) for i in range(self.n_heads)], dim=1
        )
        x = F.elu(x)
        x = self.dropout(x)
        logits = self.attention2(x, adj_mat)
        logits = self.dropout(logits)
        return logits


class GATInductive(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_classes, n_heads_1, n_heads_2):
        super(GATInductive, self).__init__()
        self.attention1 = nn.ModuleList(
            AttentionLayer(node_dim, hidden_dim) for _ in range(n_heads_1)
        )
        # only one head for second layer
        self.attention2 = nn.ModuleList(
            AttentionLayer(hidden_dim * n_heads_1, hidden_dim) for _ in range(n_heads_1)
        )
        self.attention3 = nn.ModuleList(
            AttentionLayer(hidden_dim * n_heads_1, n_classes) for _ in range(n_heads_2)
        )

    def forward(self, x, adj_mat):
        x = torch.cat([attention(x, adj_mat) for attention in self.attention1], dim=1)
        x = F.elu(x)
        residual = x
        x = torch.cat([attention(x, adj_mat) for attention in self.attention2], dim=1)
        x = F.elu(x) + residual
        logits = torch.mean(
            torch.stack([attention(x, adj_mat) for attention in self.attention3]), dim=0
        )
        return logits
