import torch.nn.functional as F

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def get_pred(model, data, edge_index):
    model.eval()
    out = model(data.x, edge_index)
    return out.argmax(dim=1)
