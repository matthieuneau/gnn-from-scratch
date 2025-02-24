"""Train script designed to work on Zinc dataset. Will make it more modular later"""

import torch
from datasets import load_dataset

from model import VanillaCGN

input_dim = 1
node_dim = 10
model = VanillaCGN(input_dim=input_dim, node_dim=node_dim, n_layers=2)

train_dataset = load_dataset("graphs-datasets/ZINC", split="train[:100]")
eval_dataset = load_dataset("graphs-datasets/ZINC", split="train[:100]")


def build_adj_mat(example):
    """Build the adjacency matrix of a graph from edge_index that looks like this: [[1, 4, 5, 6], [2, 3, 3, 5]]
    cf. https://huggingface.co/datasets/graphs-datasets/ZINC for more details"""
    edge_index = example["edge_index"]
    n_nodes = max(max(edge_index[0]), max(edge_index[1])) + 1
    n_edges = len(edge_index[0])
    adj_mat = torch.zeros((n_nodes, n_nodes))
    for i in range(n_edges):
        adj_mat[edge_index[0][i], edge_index[1][i]] = 1
    adj_mat += torch.eye(adj_mat.shape[0])  # Ensures deg_i > 0 and stabilize training
    example["adj_mat"] = adj_mat
    return example


train_dataset = train_dataset.map(build_adj_mat)


print(train_dataset[0])
