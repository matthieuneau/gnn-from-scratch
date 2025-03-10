import torch.nn as nn
import torch.nn.functional as F
import torch
import yaml
from torch_geometric.datasets import Planetoid
import torch.optim as optim

import wandb
from GAT import GAT

with open("configGAT.yaml", "r") as file:
    config = yaml.safe_load(file)

wandb.init(project="gnn-from-scratch", config=config)

node_dim = config["node_dim"]
hidden_dim = config["hidden_dim"]
batch_size = config["batch_size"]
lr = config["lr"]
n_classes = config["n_classes"]
n_epochs = config["n_epochs"]

# 7 classes in the dataset. We calibrate to reproduce the GAT paper
dataset = Planetoid(
    "./data/", "Cora", num_train_per_class=140, num_val=500, num_test=1000
)
data = dataset[0]  # there is only one graph
# One hot encoding labels for classification task
data.y = F.one_hot(data.y).float()


model = GAT(node_dim=node_dim, hidden_dim=hidden_dim, n_classes=n_classes)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=3, eta_min=1e-5
# )

for i in range(n_epochs):
    optimizer.zero_grad()
    y_pred = model(data.x, data.edge_index)
    train_loss = loss_fn(y_pred[data.train_mask], data.y[data.train_mask])
    train_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        with torch.no_grad():
            y_pred = model(data.x, data.edge_index)
            valid_loss = loss_fn(y_pred[data.val_mask], data.y[data.val_mask])
        print(
            f"Epoch {i:03d} | Train Loss: {train_loss.item():.4f} | Valid Loss: {valid_loss.item():.4f}"
        )
    wandb.log({"train_loss": train_loss.item(), "eval_loss": valid_loss.item()})
