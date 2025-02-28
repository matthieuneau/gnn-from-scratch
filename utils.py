import torch


def adj_mat_to_adj_list(adj_mat: torch.Tensor):
    edges = torch.nonzero(adj_mat, as_tuple=False)
    adj_list = [[] for _ in range(adj_mat.shape[0])]
    for i in range(len(edges)):
        u, v = edges[i]
        adj_list[v].append(u)
    return adj_list


def run_inference(model, n_samples, dataloader):
    with torch.no_grad():
        for i in range(n_samples):
            batch = next(iter(dataloader))
            y_true = batch["y"]
            y_pred = model(batch["node_feat"], batch["adj_mat"])
            print("y_true", y_true, "y_pred", y_pred)
