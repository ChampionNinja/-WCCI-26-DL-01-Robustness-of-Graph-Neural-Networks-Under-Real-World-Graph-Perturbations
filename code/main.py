import torch
from torch_geometric.datasets import Planetoid
from models import GCN, GAT, SAGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0].to(device)

model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
