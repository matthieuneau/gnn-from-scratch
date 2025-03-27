import numpy as np
import torch
from pydantic import Field
from typing_extensions import Annotated

from models import *  # noqa: F403

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, n_samples, dataloader):
    with torch.no_grad():
        for i in range(n_samples):
            batch = next(iter(dataloader))
            y_true = batch.y.to(device)
            y_pred = model(
                batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)
            )
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


def build_model(config, model_name, device):
    model = eval(model_name)(**config[model_name])
    return model.to(device)


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


def build_edge_pred_datasets(
    data,
    n_train: Annotated[float, Field(strict=True, gt=0)],
    n_val: Annotated[float, Field(strict=True, gt=0)],
    n_test: Annotated[float, Field(strict=True, gt=0)],
):
    dataset_size = data.edge_index.shape[1]
    train_size = int(n_train * dataset_size)
    val_size = int(n_val * dataset_size)

    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    train_data = data.edge_index[:, train_idx]
    val_data = data.edge_index[:, val_idx]
    test_data = data.edge_index[:, test_idx]

    return train_data, val_data, test_data


def compute_A_hat(node_features, edge_index):
    A = build_adj_mat(node_features, edge_index)
    A_tilde = A + torch.eye(A.shape[0])
    D_tilde_diagonal = torch.sum(A_tilde, axis=1)
    A_hat = (
        torch.diag(D_tilde_diagonal**-0.5)
        @ A_tilde
        @ torch.diag(D_tilde_diagonal**-0.5)
    )
    return A_hat


def build_classifier_batch(
    edge_index, node_embeddings, batch_size: int, negative_samples: int
) -> torch.Tensor:
    """Build a batch of positive and negative examples for the edge prediction task
    negative_sampling: int, number of negative examples to sample for each positive example"""

    # Start by building the positive examples
    batch_indices = np.random.choice(edge_index.shape[1], batch_size, replace=False)
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    pairs = [edge for i, edge in enumerate(edges) if i in batch_indices]
    positive_samples = torch.stack(
        [
            torch.cat([node_embeddings[pair[0]], node_embeddings[pair[1]]])
            for pair in pairs
        ]
    )

    # Add negative examples. We take them at random and don't check if they are actual edges since it is very unlikely. We allow self loops as negative examples
    random_edges = np.random.choice(
        node_embeddings.shape[0],
        (batch_size * negative_samples, 2),
        replace=True,
    )

    random_edges1 = random_edges[:, 0]
    random_edges2 = random_edges[:, 1]
    negative_samples1 = node_embeddings[random_edges1]
    negative_samples2 = node_embeddings[random_edges2]
    negative_samples = torch.cat([negative_samples1, negative_samples2], dim=1)

    batch = torch.cat([positive_samples, negative_samples], dim=0)

    return batch
