"""Train script designed to work on Zinc dataset. Will make it more modular later"""

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader

from model import VanillaCGN

# torch.set_default_dtype(torch.float32)

input_dim = 1
node_dim = 10
n_epochs = 2
train_size = 100
eval_size = 100
batch_size = 1  # TODO: Implement batch support
model = VanillaCGN(input_dim=input_dim, node_dim=node_dim, n_layers=2)

train_dataset = load_dataset("graphs-datasets/ZINC", split=f"train[:{train_size}]")
eval_dataset = load_dataset("graphs-datasets/ZINC", split=f"validation[:{eval_size}]")


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


def to_float_tensor(example):
    return {
        "node_feat": torch.tensor(example["node_feat"], dtype=torch.float32).tolist(),
        "adj_mat": torch.tensor(example["adj_mat"], dtype=torch.float32).tolist(),
        "y": torch.tensor(example["y"], dtype=torch.float32).tolist(),
    }


train_dataset = train_dataset.map(build_adj_mat)
train_dataset = train_dataset.select_columns(["node_feat", "adj_mat", "y"])
eval_dataset = eval_dataset.map(build_adj_mat)
eval_dataset = eval_dataset.select_columns(["node_feat", "adj_mat", "y"])


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def collate_fn(batch):
    """Convert dataset columns to torch.float32 inside the DataLoader."""
    return {
        "node_feat": torch.stack(
            [torch.tensor(b["node_feat"], dtype=torch.float32) for b in batch]
        ),
        "adj_mat": torch.stack(
            [torch.tensor(b["adj_mat"], dtype=torch.float32) for b in batch]
        ),
        "y": torch.stack([torch.tensor(b["y"], dtype=torch.float32) for b in batch]),
    }


train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

batch = next(iter(train_dataloader))

for i in range(n_epochs):
    total_loss = 0
    for j, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        # print(batch["node_feat"])
        # print(batch["adj_mat"])
        # exit()
        y_pred = model(batch["node_feat"], batch["adj_mat"])
        y_true = batch["y"]
        loss = loss_fn(y_pred, y_true)
        total_loss += loss
        loss.backward()
        optimizer.step()
        exit()
    total_loss /= len(train_dataset)
    print(f"loss {total_loss} at epoch {i}")


# print(train_dataset[0])
