import datasets
from torch_geometric.datasets import Planetoid
from utils import build_adj_mat

# 7 classes in the dataset. We calibrate to reproduce the GAT paper
dataset = Planetoid(
    "./data/", "Cora", num_train_per_class=140, num_val=500, num_test=1000
)

print(dataset[0])
print(build_adj_mat(dataset[0].x, dataset[0].edge_index))
