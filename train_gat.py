import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import yaml
from torch_geometric.datasets import Planetoid
import torch.optim as optim

import wandb
from GAT import GAT
from utils import build_adj_mat

with open("configGAT.yaml", "r") as file:
    config = yaml.safe_load(file)

wandb.init(project="gnn-from-scratch", config=config)

node_dim = config["node_dim"]
hidden_dim = config["hidden_dim"]
batch_size = config["batch_size"]
lr = config["lr"]
n_classes = config["n_classes"]
n_epochs = config["n_epochs"]
n_train = config["n_train"]
n_val = config["n_val"]
n_test = config["n_test"]
dropout = config["dropout"]

# 7 classes in the dataset. We calibrate to reproduce the GAT paper
dataset = Planetoid(
    "./data/", "Cora", num_train_per_class=n_train, num_val=n_val, num_test=n_test
)
data = dataset[0]  # there is only one graph
# One hot encoding labels for classification task
data.y = F.one_hot(data.y).float()
data.adj_mat = build_adj_mat(data.x, data.edge_index)

print(data.adj_mat.shape)

model = GAT(
    node_dim=node_dim, hidden_dim=hidden_dim, n_classes=n_classes, dropout=dropout
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=3, eta_min=1e-5
# )

for i in range(n_epochs):
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
