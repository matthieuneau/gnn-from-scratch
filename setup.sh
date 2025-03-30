set -e

apt update
apt install python3.10-venv
git clone https://github.com/matthieuneau/gnn-from-scratch.git
cd gnn-from-scratch
git switch edge-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
wandb login d83092c0f0a93b3039ddc492036fb63f7a1326b0
python3 gat_inductive.py