import torch.nn.functional as F
import torch.nn as nn
import torch

from utils import build_adj_mat


class GAT(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_classes):
        super(GAT, self).__init__()
        self.att1 = AttentionLayer(
            node_dim, hidden_dim
        )  # 1433 is node dim, 3000 is hidden dim
        self.fc3 = nn.Linear(
            hidden_dim, n_classes
        )  # 7 classes for each node, 50 input dim

    def forward(self, x, edge_index):
        adj_mat = build_adj_mat(x, edge_index)
        x = self.att1(x, adj_mat)
        logits = self.fc3(x)
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
