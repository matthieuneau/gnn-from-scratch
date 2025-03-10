import torch.nn.functional as F
import torch.nn as nn
import torch


class GAT(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_classes, n_heads, dropout):
        super(GAT, self).__init__()
        self.n_heads = n_heads
        self.attention = nn.ModuleList(
            AttentionLayer(node_dim, hidden_dim) for _ in range(n_heads)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * n_heads, n_classes)

    def forward(self, x, adj_mat):
        x = torch.cat(
            [self.attention[i](x, adj_mat) for i in range(self.n_heads)], dim=1
        )
        x = self.dropout(x)
        logits = self.fc1(x)
        logits = self.dropout(logits)
        return logits


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.init.xavier_uniform_(torch.empty(output_dim, input_dim))
        self.a1 = nn.init.xavier_uniform_(
            torch.empty(output_dim, 1)
        )  # a = concat(a1, a2)
        self.a2 = nn.init.xavier_uniform_(torch.empty(output_dim, 1))

    def forward(self, x, adj_mat):
        x = x @ self.W.T  # matrix containing the W*hi vectors
        f1 = (x @ self.a1).squeeze()
        f2 = (x @ self.a2).squeeze()
        e = F.leaky_relu_(f1.unsqueeze(1) + f2.unsqueeze(0))  # Broadcast
        e = e * adj_mat  # ensures that the softmax is only performed over the neighbors
        e = torch.where(e == 0, -torch.inf, e)
        alpha = F.softmax(e, dim=1)  # or dim=0?
        x = F.elu_(alpha @ x)
        return x
