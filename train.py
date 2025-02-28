"""Train script designed to work on Zinc dataset. Will make it more modular later"""

import torch.nn.functional as F
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from model import VanillaCGN
from utils import run_inference

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

wandb.init(project="gnn-from-scratch", config=config)

input_dim = config["input_dim"]
node_dim = config["node_dim"]
n_epochs = config["n_epochs"]
train_size = config["train_size"]
eval_size = config["eval_size"]
batch_size = config["batch_size"]  # TODO: Implement batch support
n_layers = config["n_layers"]
lr = float(config["lr"])

model = VanillaCGN(input_dim=input_dim, node_dim=node_dim, n_layers=n_layers)

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


def padding(example, max_nodes):  # TODO: Add mean support
    """To be applied after builing the adj mat.
    # 'zeros' pads the node_feat matrix with zero vectors
    # 'mean' pads the node_feat matrix with vectors equal to the mean of the nodes embeddings
    The adj mat is always padded with zeros, except ones on the diagonal for stability when dividing by deg_i in the conv layer
    """
    node_dim = len(example["node_feat"][0])
    n_nodes = len(example["node_feat"])
    zeros = [0 for _ in range(node_dim)]
    # print('zeros', zeros)
    example["node_feat"] += [zeros for _ in range(max_nodes - n_nodes)]
    example["adj_mat"] = torch.tensor(example["adj_mat"])
    pad = (0, max_nodes - n_nodes, 0, max_nodes - n_nodes)
    example["adj_mat"] = F.pad(example["adj_mat"], pad, "constant", 0)
    example["adj_mat"].diagonal().fill_(1)
    example["adj_mat"] = example["adj_mat"].tolist()
    return example


max_nodes = max(max(train_dataset["num_nodes"]), max(eval_dataset["num_nodes"]))
train_dataset = train_dataset.map(build_adj_mat)
train_dataset = train_dataset.map(lambda example: padding(example, max_nodes))
train_dataset = train_dataset.select_columns(["node_feat", "adj_mat", "y"])
eval_dataset = eval_dataset.map(build_adj_mat)
eval_dataset = eval_dataset.select_columns(["node_feat", "adj_mat", "y"])
eval_dataset = eval_dataset.map(lambda example: padding(example, max_nodes))


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=3, eta_min=1e-5
# )


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


for i in tqdm(range(n_epochs)):
    train_loss = 0
    # print("lr: ", scheduler.get_lr())
    for j, batch in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        y_pred = model(batch["node_feat"], batch["adj_mat"])
        y_true = batch["y"]
        loss = loss_fn(y_pred, y_true)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)  # or divide by len(dataset) ??

    with torch.no_grad():
        eval_loss = 0
        for j, batch in tqdm(enumerate(eval_dataloader)):
            y_pred = model(batch["node_feat"], batch["adj_mat"])
            y_true = batch["y"]
            eval_loss += loss_fn(y_pred, y_true)
        eval_loss /= len(eval_dataloader)  # or divide by len(dataset) ??

    # scheduler.step()
    wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "epoch": i})

    print(f"train loss {train_loss} at epoch {i}")
    print(f"valid loss {eval_loss} at epoch {i}")
    run_inference(model, 5, train_dataloader)
