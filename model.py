import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAT_reg_class(nn.Module):
    def __init__(
        self,
        input_dim,
        node_dim,
        hidden_dim,
        output_dim,
        n_heads,
        n_hidden_layers_mlp,
        dropout,
        task,
    ):
        super().__init__()
        if task not in [
            "regression",
            "classification",
            "node_classification",
            "link_prediction",
        ]:
            raise ValueError(
                "task must be either 'regression', 'classification', 'node_classification' or 'link_prediction'"
            )
        self.task = task
        self.n_heads = n_heads
        self.U0 = nn.Linear(input_dim, node_dim, bias=False)
        self.attention = nn.ModuleList(
            AttentionLayer(node_dim, hidden_dim) for _ in range(n_heads)
        )
        self.dropout = nn.Dropout(dropout)
        in_dim_MLP = (
            hidden_dim * n_heads
            if task in ["classification", "regression", "node_classification"]
            else 2 * hidden_dim * n_heads
        )
        self.readOutLayer = MLPReadout(
            in_dim_MLP, output_dim, n_hidden_layers_mlp, hidden_dim, bias=True
        )

    def forward(self, x, edge_index, batch=None):
        adj_mat = build_adj_mat(edge_index)
        x = x.float()
        x = self.U0(x)
        x = torch.cat(
            [self.attention[i](x, adj_mat) for i in range(self.n_heads)], dim=1
        )
        x = self.dropout(x)
        if self.task in ["classification", "regression"]:
            x = batch_mean_pool(x, batch)
        elif self.task == "node_classification":
            x = x
        elif self.task == "link_prediction":
            x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        logits = self.readOutLayer(x)
        logits = self.dropout(logits)
        return logits


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device=device):
        super().__init__()
        self.W = nn.init.xavier_uniform_(torch.empty(output_dim, input_dim)).to(device)
        self.a1 = nn.init.xavier_uniform_(torch.empty(output_dim, 1)).to(
            device
        )  # a = concat(a1, a2)
        self.a2 = nn.init.xavier_uniform_(torch.empty(output_dim, 1)).to(device)

    def forward(self, x, adj_mat):
        x = x @ self.W.T  # matrix containing the W*hi vectors
        f1 = (x @ self.a1).squeeze()
        f2 = (x @ self.a2).squeeze()
        e = F.leaky_relu(f1.unsqueeze(1) + f2.unsqueeze(0))  # Broadcast
        e = e * adj_mat  # ensures that the softmax is only performed over the neighbors
        e = torch.where(e == 0, torch.tensor(-1e9, device=e.device), e)
        alpha = F.softmax(e, dim=1)  # or dim=0?
        x = F.elu(alpha @ x)
        return x


class GCN_reg_class(nn.Module):
    def __init__(
        self,
        input_dim,
        n_layers_conv,
        node_dim,
        n_hidden_layers_mlp,
        hidden_dim,
        output_dim,
        task,
    ) -> None:
        super().__init__()
        if task not in [
            "regression",
            "classification",
            "node_classification",
            "link_prediction",
        ]:
            raise ValueError(
                "task must be either 'regression', 'classification', 'node_classification' or 'link_prediction'"
            )
        self.task = task
        self.n_layers_conv = n_layers_conv
        self.U0 = nn.Linear(input_dim, node_dim, bias=False)
        self.convLayers = nn.ModuleList(
            [
                ConvNetLayer(node_dim, bias=False, dropout=0.5, residual=True)
                for _ in range(n_layers_conv)
            ]
        )
        in_dim_MLP = (
            node_dim
            if task in ["classification", "regression", "node_classification"]
            else 2 * node_dim
        )
        self.readOutLayer = MLPReadout(
            in_dim_MLP, output_dim, n_hidden_layers_mlp, hidden_dim, bias=True
        )

    def forward(self, x, edge_index, batch=None):
        # print(x.shape)
        x = x.float()
        x = self.U0(x)  # self.b0 is broadcasted properly?
        # print(x.shape)
        for i in range(self.n_layers_conv):
            x = self.convLayers[i](x, edge_index)
        if self.task in ["classification", "regression"]:
            x = batch_mean_pool(x, batch)
        elif self.task == "node_classification":
            x = x
        elif self.task == "link_prediction":
            x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        # print(x.shape)
        x = self.readOutLayer(x)
        # print(x.shape)
        return x


class ConvNetLayer(nn.Module):
    def __init__(self, node_dim, bias=False, dropout=0.5, residual=True) -> None:
        super().__init__()
        self.U = nn.Linear(node_dim, node_dim, bias=bias)
        self.batchnorm = nn.BatchNorm1d(node_dim)
        self.activation = nn.Tanh()  # non linear activation function
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, edge_index):
        x_in = x
        num_nodes = x.shape[0]
        diffusion_matrix = build_diffusion_matrix(edge_index, num_nodes=num_nodes)
        x = diffusion_matrix @ x
        x = self.U(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        if self.residual:
            x = x + x_in
        x = self.dropout(x)
        return x


class MLPReadout(nn.Module):
    def __init__(
        self, input_dim, output_dim, n_hidden_layers_mlp, hidden_dim, bias=True
    ):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim, hidden_dim, bias=bias)]
        list_FC_layers += [
            nn.Linear(hidden_dim, hidden_dim, bias=bias)
            for l in range(n_hidden_layers_mlp)
        ]
        list_FC_layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = n_hidden_layers_mlp

    def forward(self, x):
        y = x
        for l in range(self.L + 1):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L + 1](y).squeeze()
        return y


def build_adj_mat(edge_index, num_nodes=None):
    """
    Build sparse adjacency matrix with self-loops.
    Args:
        edge_index: Tensor of shape (2, E)
        num_nodes: Optional number of nodes (int)
    Returns:
        Sparse adjacency matrix (num_nodes x num_nodes)
    """
    if num_nodes is None:
        num_nodes = int(torch.max(edge_index)) + 1

    # Ajouter les self-loops : concaténer les indices diagonaux
    self_loops = torch.arange(num_nodes, device=edge_index.device)
    self_loop_edges = torch.stack([self_loops, self_loops], dim=0)

    edge_index_with_loops = torch.cat([edge_index, self_loop_edges], dim=1)
    values = torch.ones(edge_index_with_loops.size(1), device=edge_index.device)

    # Créer la matrice d'adjacence sparse
    adj = torch.sparse_coo_tensor(edge_index_with_loops, values, (num_nodes, num_nodes))
    return adj.coalesce()  # Important pour avoir des indices triés


def build_diffusion_matrix(edge_index, num_nodes=None):
    """
    Compute the normalized diffusion matrix: D^{-1/2} A D^{-1/2}
    Returns a sparse matrix.
    """
    adj = build_adj_mat(edge_index, num_nodes)

    # Calcul du degré (somme par ligne)
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # Shape (N,)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0  # Éviter les inf

    # Construire D^{-1/2} * A * D^{-1/2}
    row, col = adj.indices()
    values = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    diffusion = torch.sparse_coo_tensor(adj.indices(), values, adj.size())
    return diffusion.coalesce()


def batch_mean_pool(x, batch):
    """Effectue une moyenne des features par graphe"""
    batch_size = batch.max().item() + 1
    mean_pooled = torch.zeros((batch_size, x.shape[1]), device=x.device)
    count = torch.zeros(batch_size, device=x.device)

    for i in range(batch.shape[0]):
        mean_pooled[batch[i]] += x[i]
        count[batch[i]] += 1

    mean_pooled /= count.unsqueeze(1)  # Normalisation
    return mean_pooled
