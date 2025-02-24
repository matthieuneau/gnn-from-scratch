"""Train script designed to work on Zinc dataset. Will make it more modular later"""

from model import VanillaCGN

input_dim = 1
node_dim = 20
model = VanillaCGN(input_dim=input_dim, node_dim=node_dim, n_layers=4)
