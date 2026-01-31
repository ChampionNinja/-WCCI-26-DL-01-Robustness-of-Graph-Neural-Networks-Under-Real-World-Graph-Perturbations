import csv
from train import train, get_pred
from metrics import accuracy, stability
from perturbations import random_removal, degree_aware_removal, class_aware_removal

def run_experiment(model, data, optimizer, levels, seeds, name, writer):
    clean_pred = get_pred(model, data, data.edge_index)

    for p in levels:
        for pert in ["random", "degree", "class"]:
            if pert == "random":
                pe = random_removal(data.edge_index, p)
            elif pert == "degree":
                pe = degree_aware_removal(data.edge_index, data.num_nodes, p)
            else:
                pe = class_aware_removal(data.edge_index, data.y, p)

            pert_pred = get_pred(model, data, pe)
            acc = accuracy(pert_pred, data.y, data.test_mask)
            stab = stability(clean_pred, pert_pred, data.test_mask)

            writer.writerow([name, pert, p, acc, stab])
