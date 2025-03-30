import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch_geometric.datasets import Planetoid
from torchinfo import summary
from tqdm import tqdm

import wandb
from models import GATTransductive
from utils import build_adj_mat

with open("configGATTransductive.yaml", "r") as file:
    config = yaml.safe_load(file)

wandb.init(project="gnn-from-scratch", config=config)

node_dim = config["node_dim"]
hidden_dim = config["hidden_dim"]
lr = config["lr"]
n_classes = config["n_classes"]
batch_size = config["batch_size"]
n_epochs = config["n_epochs"]
dropout = config["dropout"]
n_heads = config["n_heads"]
weight_decay = config["weight_decay"]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 7 classes in the dataset. We calibrate to reproduce the GAT paper
dataset = Planetoid(root="./data/", name="Cora")
n_train = dataset[0].train_mask.sum().item()
n_val = dataset[0].val_mask.sum().item()
n_test = dataset[0].test_mask.sum().item()
data = dataset[0].to(device)  # there is only one graph
data.adj_mat = build_adj_mat(data.x, data.edge_index, device)

print(data.adj_mat.shape)

model = GATTransductive(
    node_dim=node_dim,
    hidden_dim=hidden_dim,
    n_classes=n_classes,
    n_heads=n_heads,
    dropout=dropout,
).to(device)
model_summary = summary(model)
wandb.config.update({"total_params": model_summary.total_params})

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

for i in tqdm(range(n_epochs)):
    model.train()
    batch = np.random.choice(np.arange(n_train), size=batch_size, replace=False)
    batch_mask = torch.zeros_like((data.train_mask))
    batch_mask[batch] = True

    optimizer.zero_grad()
    y_pred = model(data.x, data.adj_mat)
    train_loss = loss_fn(y_pred[batch_mask], data.y[batch_mask])
    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        y_pred = model(data.x, data.adj_mat)[data.val_mask]
        y_true = data.y[data.val_mask]
        valid_loss = loss_fn(y_pred, y_true)
        labels_pred = torch.argmax(y_pred, dim=1)
        labels = torch.argmax(y_true, dim=1)
        accuracy = torch.sum(labels == labels_pred) / n_val

    if i % 10 == 0:
        print(
            f"Epoch {i:03d} | Train Loss: {train_loss.item():.4f} | Valid Loss: {valid_loss.item():.4f} | Accuracy: {accuracy:.4f}"
        )

    wandb.log(
        {
            "train_loss": train_loss.item(),
            "eval_loss": valid_loss.item(),
            "accuracy": accuracy,
        }
    )
