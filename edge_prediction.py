import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

import wandb
from utils import (
    build_adj_mat,
    build_classifier_batch,
    build_edge_pred_datasets,
    build_model,
    compute_A_hat,
)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

wandb.init(project="gnn-from-scratch", config=config)
# save config and current file
wandb.save(__file__)
wandb.save("models.py")

negative_samples_factor = config["negative_samples_factor"]
batch_size = config["batch_size"]
lr = config["lr"]
n_epochs = config["n_epochs"]
n_train = config["n_train"]
n_val = config["n_val"]
n_test = config["n_test"]
weight_decay = config["weight_decay"]
dataset_name = config["dataset"]
hits_k_rank = config["HITS@K_rank"]
hits_k_positive_samples = config["HITS@K_positive_samples"]
hits_k_negative_samples = config["HITS@K_negative_samples_factor"]

# Keep device to cpu for now because we don't use dataloaders properly so it's faster that way
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# device = "cpu"

dataset = Planetoid(
    "./data/", dataset_name, num_train_per_class=n_train, num_val=n_val, num_test=n_test
)

data = dataset[0]  # there is only one graph
data.to(device)
# # One hot encoding labels for classification task
# data.y = F.one_hot(data.y).float()
# data.adj_mat = build_adj_mat(data.x, data.edge_index)

train_edge_index, val_edge_index, test_edge_index = build_edge_pred_datasets(
    data, n_train, n_val, n_test
)

model = build_model(config, "GCN", device)
classifier = build_model(config, "EdgePrediction", device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer_model = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_classifier = optim.Adam(
    classifier.parameters(), lr=lr, weight_decay=weight_decay
)
scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_model, mode="min", factor=0.2, patience=10, verbose=True
)
scheduler_classifier = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_classifier, mode="min", factor=0.2, patience=10, verbose=True
)

if model.__class__.__name__ == "GCN":
    data.A_hat = compute_A_hat(data.x, data.edge_index).to(device)

if model.__class__.__name__ == "GAT":
    data.adj_mat = build_adj_mat(data.x, data.edge_index).to(device)

for i in tqdm(range(n_epochs)):
    model.train()
    classifier.train()
    optimizer_classifier.zero_grad()
    optimizer_model.zero_grad()

    # No matter what edges will be compared, apply the model to the whole graph
    node_embeddings = model(data)

    # TODO: Inefficient for now. This recomputes batch indices at each iteration
    batch = build_classifier_batch(
        train_edge_index, node_embeddings, batch_size, negative_samples_factor
    ).to(device)
    labels = torch.cat(
        [torch.ones(batch_size), torch.zeros(batch_size * negative_samples_factor)]
    ).to(device)

    # Shuffle the batch to mix positive and negative examples
    perm = torch.randperm(batch.size(0))
    batch, labels = batch[perm], labels[perm]

    logits = classifier(batch)
    train_loss = loss_fn(logits, labels.squeeze())

    train_loss.backward()
    optimizer_classifier.step()
    optimizer_model.step()
    scheduler_model.step(train_loss)
    scheduler_classifier.step(train_loss)

    if i % 10 == 0:
        with torch.no_grad():
            model.eval()
            classifier.eval()
            node_embeddings = model(data).to(device)
            batch = build_classifier_batch(
                val_edge_index, node_embeddings, batch_size, negative_samples_factor
            ).to(device)
            labels = torch.cat(
                [
                    torch.ones(batch_size, device=device),
                    torch.zeros(batch_size * negative_samples_factor, device=device),
                ]
            )

            logits = classifier(batch)
            val_loss = loss_fn(logits.reshape(-1), labels.reshape(-1))

            # TODO: augment the number of hitsK tested
            hits_k_batch = build_classifier_batch(
                val_edge_index,
                node_embeddings,
                hits_k_positive_samples,
                hits_k_negative_samples,
            )

            logits = classifier(hits_k_batch)
            # Use the double argsort trick to get the ranks
            ranks = torch.argsort(torch.argsort(logits, descending=True)) + 1
            positive_samples_ranks = ranks[:hits_k_positive_samples]
            val_hits_k_accuracy = torch.mean(
                (positive_samples_ranks <= hits_k_rank).float()
            ).item()

        print(
            f"Epoch {i:03d} | Train Loss: {train_loss.item():.4f} | Valid Loss: {val_loss.item():.4f} | HITS@{hits_k_rank} accuracy: {val_hits_k_accuracy:.4f}"
        )

        wandb.log(
            {
                "train_loss": train_loss.item(),
                "eval_loss": val_loss.item(),
                "HITS@k": val_hits_k_accuracy,
                "model_lr": optimizer_model.param_groups[0]["lr"],
                "classifier_lr": optimizer_classifier.param_groups[0]["lr"],
            },
            step=i,
        )
