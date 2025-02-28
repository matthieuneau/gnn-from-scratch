import torch
import yaml
from torchinfo import summary

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


adj_mat = torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]])


def adj_mat_to_adj_list(adj_mat: torch.Tensor):
    edges = torch.nonzero(adj_mat, as_tuple=False)
    adj_list = [[] for _ in range(adj_mat.shape[0])]
    for i in range(len(edges)):
        u, v = edges[i]
        adj_list[v].append(u)
    return adj_list


print(adj_mat_to_adj_list(adj_mat))
