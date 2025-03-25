import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

import wandb
from GCN import GCN, EdgePrediction
from utils import (
    build_adj_mat,
    build_classifier_batch,
    build_edge_pred_datasets,
    compute_A_hat,
)

with open("configEdgePred.yaml", "r") as file:
    config = yaml.safe_load(file)
    config = config["GCN"]

wandb.init(project="gnn-from-scratch", config=config)

negative_samples_factor = config["negative_samples_factor"]
node_dim = config["node_dim"]
hidden_dim = config["hidden_dim"]
batch_size = config["batch_size"]
lr = config["lr"]
n_epochs = config["n_epochs"]
n_train = config["n_train"]
n_val = config["n_val"]
n_test = config["n_test"]
dropout = config["dropout"]
weight_decay = config["weight_decay"]
dataset_name = config["dataset"]
hits_k_rank = config["HITS@K_rank"]
hits_k_positive_samples = config["HITS@K_positive_samples"]
hits_k_negative_samples = config["HITS@K_negative_samples_factor"]

dataset = Planetoid(
    "./data/", dataset_name, num_train_per_class=n_train, num_val=n_val, num_test=n_test
)

data = dataset[0]  # there is only one graph
# # One hot encoding labels for classification task
# data.y = F.one_hot(data.y).float()
# data.adj_mat = build_adj_mat(data.x, data.edge_index)

train_edge_index, val_edge_index, test_edge_index = build_edge_pred_datasets(
    data, n_train, n_val, n_test
)

gcn = GCN(
    input_dim=node_dim,
    hidden_dim=hidden_dim,
    output_dim=hidden_dim,
    n_layers=3,
    dropout=dropout,
)

edge_pred = EdgePrediction(embedding_dim=hidden_dim)

loss_fn = nn.BCEWithLogitsLoss()
optimizer_gcn = optim.Adam(gcn.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_edge_pred = optim.Adam(
    edge_pred.parameters(), lr=lr, weight_decay=weight_decay
)

data.A_hat = compute_A_hat(data.x, data.edge_index)

for i in tqdm(range(n_epochs)):
    gcn.train()
    edge_pred.train()
    optimizer_edge_pred.zero_grad()
    optimizer_gcn.zero_grad()

    # No matter what edges will be compared, apply the GCN to the whole graph
    node_embeddings = gcn(data.x, data.A_hat)

    # TODO: Inefficient for now. This recomputes batch indices at each iteration
    batch = build_classifier_batch(
        train_edge_index, node_embeddings, batch_size, negative_samples_factor
    )
    labels = torch.cat(
        [torch.ones(batch_size), torch.zeros(batch_size * negative_samples_factor)]
    )

    # Shuffle the batch to mix positive and negative examples
    perm = torch.randperm(batch.size(0))
    batch, labels = batch[perm], labels[perm]

    # TODO: shuffle the batch to mix positive and negative examples
    logits = edge_pred(batch)
    train_loss = loss_fn(logits, labels.squeeze())

    train_loss.backward()
    optimizer_edge_pred.step()
    optimizer_gcn.step()

    with torch.no_grad():
        gcn.eval()
        edge_pred.eval()
        node_embeddings = gcn(data.x, data.A_hat)
        batch = build_classifier_batch(
            val_edge_index, node_embeddings, batch_size, negative_samples_factor
        )
        labels = torch.cat(
            [torch.ones(batch_size), torch.zeros(batch_size * negative_samples_factor)]
        )

        logits = edge_pred(batch)
        val_loss = loss_fn(logits.reshape(-1), labels.reshape(-1))

        # TODO: augment the number of hitsK tested
        hits_k_batch = build_classifier_batch(
            val_edge_index,
            node_embeddings,
            hits_k_positive_samples,
            hits_k_negative_samples,
        )

        logits = edge_pred(hits_k_batch)
        # Use the double argsort trick to get the ranks
        ranks = torch.argsort(torch.argsort(logits, descending=True)) + 1
        positive_samples_ranks = ranks[:hits_k_positive_samples]
        hits_k_accuracy = torch.mean(
            (positive_samples_ranks <= hits_k_rank).float()
        ).item()

    if i % 10 == 0:
        print(
            f"Epoch {i:03d} | Train Loss: {train_loss.item():.4f} | Valid Loss: {val_loss.item():.4f} | HITS@{hits_k_rank} accuracy: {hits_k_accuracy:.4f}"
        )

    wandb.log(
        {
            "train_loss": train_loss.item(),
            "eval_loss": val_loss.item(),
            "accuracy": hits_k_accuracy,
        }
    )
