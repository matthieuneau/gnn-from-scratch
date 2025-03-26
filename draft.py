import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from GAT_1 import GAT
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

import wandb
from models import GCN, EdgePrediction
from utils import (
    build_adj_mat,
    build_classifier_batch,
    build_edge_pred_datasets,
    build_model,
    compute_A_hat,
)

with open("configEdgePred.yaml", "r") as file:
    config = yaml.safe_load(file)
    # config = config["GCN"]

wandb.init(project="gnn-from-scratch", config=config)
# save config and current file
wandb.save("configEdgePred.yaml", policy="now")
# wandb.save("GCN.py")
wandb.save("GAT.py")


# Keep device to cpu for now because we don't use dataloaders properly so it's faster that way
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = build_model(config)

model.to(device)

print(model)
