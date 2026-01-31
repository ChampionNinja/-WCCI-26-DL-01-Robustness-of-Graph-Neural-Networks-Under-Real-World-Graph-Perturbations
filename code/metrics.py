def accuracy(pred, y, mask):
    return (pred[mask] == y[mask]).sum().item() / mask.sum().item()

def stability(pred_clean, pred_pert, mask):
    return (pred_clean[mask] == pred_pert[mask]).sum().item() / mask.sum().item()
