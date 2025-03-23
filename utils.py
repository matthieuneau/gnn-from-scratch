import torch


def run_inference(model, n_samples, dataloader):
    with torch.no_grad():
        for i in range(n_samples):
            batch = next(iter(dataloader))
            y_true = batch["y"]
            y_pred = model(batch["node_feat"], batch["adj_mat"])
            print("y_true", y_true, "y_pred", y_pred)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def build_adj_mat(node_features, edge_index):
    """Build the adjacency matrix of a graph from edge_index that looks like this: [[1, 4, 5, 6], [2, 3, 3, 5]]
    cf. https://huggingface.co/datasets/graphs-datasets/ZINC for more details"""
    n_nodes = node_features.shape[0]
    n_edges = len(edge_index[0])
    adj_mat = torch.zeros((n_nodes, n_nodes))
    for i in range(n_edges):
        adj_mat[edge_index[0][i], edge_index[1][i]] = 1
    adj_mat += torch.eye(adj_mat.shape[0])  # Ensures deg_i > 0 and stabilize training
    return adj_mat


# Build the adjacency matrices for the dataset
def build_adj_mat_hashmap(train_dataset, val_dataset, test_dataset):
    adjacency_matrices = {"train": [], "val": [], "test": []}

    for data in train_dataset:
        adj_mat = build_adj_mat(data.x, data.edge_index)
        adjacency_matrices["train"].append(adj_mat)

    for data in val_dataset:
        adj_mat = build_adj_mat(data.x, data.edge_index)
        adjacency_matrices["val"].append(adj_mat)

    for data in test_dataset:
        adj_mat = build_adj_mat(data.x, data.edge_index)
        adjacency_matrices["test"].append(adj_mat)

    return adjacency_matrices
