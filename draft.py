"""Train script designed to work on Zinc dataset. Will make it more modular later"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from model import VanillaCGN

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

input_dim = config["input_dim"]
node_dim = config["node_dim"]
n_epochs = config["n_epochs"]
train_size = config["train_size"]
eval_size = config["eval_size"]
batch_size = config["batch_size"]  # TODO: Implement batch support
n_layers = config["n_layers"]
model = VanillaCGN(input_dim=input_dim, node_dim=node_dim, n_layers=n_layers)
summary(model)
exit()
