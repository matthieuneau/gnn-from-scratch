import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.datasets import PPI
from torchmetrics.classification import MultilabelF1Score
from tqdm import tqdm

import wandb
from GAT import GATInductive
from utils import build_adj_mat_hashmap

with open("configGAT.yaml", "r") as file:
    config = yaml.safe_load(file)
    config = config["PPI"]

wandb.init(project="gnn-from-scratch", config=config)

lr = config["lr"]
node_dim = config["node_dim"]
n_classes = config["n_classes"]
n_epochs = config["n_epochs"]
n_train = config["n_train"]
n_val = config["n_val"]
n_test = config["n_test"]
batch_size = config["batch_size"]
hidden_dim = config["hidden_dim"]
n_heads_1 = config["n_heads_1"]
n_heads_2 = config["n_heads_2"]

train_dataset = PPI("./data/", split="train")
val_dataset = PPI("./data/", split="val")
test_dataset = PPI("./data/", split="test")

model = GATInductive(
    node_dim=node_dim,
    hidden_dim=hidden_dim,
    n_classes=n_classes,
    n_heads_1=n_heads_1,
    n_heads_2=n_heads_2,
)

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCEWithLogitsLoss()

adj_mat_hashmap = build_adj_mat_hashmap(train_dataset, val_dataset, test_dataset)

num_labels = train_dataset[0].y.shape[1]
f1_metric = MultilabelF1Score(num_labels=num_labels, average="micro")

for i in tqdm(range(n_epochs)):
    model.train()
    optimizer.zero_grad()
    index = np.random.choice(np.arange(n_train))  # TODO: handle batch_size >1
    data = train_dataset[index]

    y_pred = model(data.x, adj_mat_hashmap["train"][index])
    train_loss = loss_fn(y_pred, data.y)
    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        index = np.random.choice(np.arange(n_val))
        data = val_dataset[index]

        y_pred = model(data.x, adj_mat_hashmap["val"][index])

        y_true = data.y
        valid_loss = loss_fn(y_pred, y_true)
        labels_pred = (y_pred > 0).int()
        f1 = f1_metric(labels_pred, y_true)

    if i % 20 == 0:
        print(
            f"Epoch {i:03d} | Train Loss: {train_loss.item():.4f} | Valid Loss: {valid_loss.item():.4f} | F1-score: {f1:.4f}"
        )

    wandb.log(
        {
            "train_loss": train_loss.item(),
            "eval_loss": valid_loss.item(),
            "f1_score": f1,
        }
    )
