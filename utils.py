import torch


def run_inference(model, n_samples, dataloader):
    with torch.no_grad():
        for i in range(n_samples):
            batch = next(iter(dataloader))
            y_true = batch["y"]
            y_pred = model(batch["node_feat"], batch["adj_mat"])
            print("y_true", y_true, "y_pred", y_pred)
