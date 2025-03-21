import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def run_inference(model, n_samples, dataloader):
    with torch.no_grad():
        for i in range(n_samples):
            batch = next(iter(dataloader))
            y_true = batch.y.to(device)
            y_pred = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
            print("y_true", y_true, "y_pred", y_pred)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]