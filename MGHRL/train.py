import sys
import time
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx

from dhg import Graph, Hypergraph
from dhg.data import *
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from AGHG import generate_hypergraph_from_graph
from models.MGHN import MGHN


def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    train_accuracy = evaluator.validate(lbls, outs)
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}, Train Accuracy: {train_accuracy:.5f}")
    torch.cuda.empty_cache()
    return loss.item()

@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res

def generate_masks(num_v, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Generate training, validation, and test masks.

    Parameters:
    num_v (int): The number of vertices in the new hypergraph.
    train_ratio (float): The ratio for the training set. Default is 0.6.
    val_ratio (float): The ratio for the validation set. Default is 0.2.
    test_ratio (float): The ratio for the test set. Default is 0.2.

    Returns:
    tuple: A tuple containing train_mask, val_mask, and test_mask.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of ratios must be 1"
    assert num_v > 0, "The number of vertices must be greater than 0"

    indices = np.random.permutation(num_v)
    train_size = int(train_ratio * num_v)
    val_size = int(val_ratio * num_v)
    test_size = num_v - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = torch.zeros(num_v, dtype=torch.bool)
    val_mask = torch.zeros(num_v, dtype=torch.bool)
    test_mask = torch.zeros(num_v, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

def read_data(data_name):# Need to modify
    if data_name == "Cora":
        return Cora()
    elif data_name == "Pubmed":
        return Pubmed()
    elif data_name == "Citeseer":
        return Citeseer()
    elif data_name == "Facebook":
        return Facebook()
    elif data_name == "BlogCatalog":
        return BlogCatalog()
    elif data_name == "Flickr":
        return Flickr()
    elif data_name == "Github":
        return Github()
    else:
        print("data_name error")
        sys.exit()
        
import torch


if __name__ == "__main__":

    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="Cora",
                        help='name of dataset (default: Cora)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--method', type = str, default = "khop",
                        help='method')  
    parser.add_argument('--seed', type = int, default = "2001",
                        help='random seed')                       
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    
    data = read_data(args.data_name)
    print("Input", args.data_name, "loaded", type(data).__name__)

    G = Graph(data["num_vertices"], data["edge_list"])
    try:
        X, lbl = data['features'], data['labels']
        print("Dataset has features")
        if X is None or len(X) == 0:
            raise ValueError("Features are empty or None.")
    except Exception as e:
        X, lbl = torch.eye(data["num_vertices"]), data["labels"]
        print("Dataset has no features")

    if args.method == "khop":
        HG = Hypergraph.from_graph(G)
        HG.add_hyperedges_from_graph_kHop(G, k=1)
    elif args.method == "MGHRL":
        HG = generate_hypergraph_from_graph(G)

    # Generate masks with fixed ratios
    train_mask, val_mask, test_mask = generate_masks(G.num_v, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    print(f"Training set size: {train_mask.sum().item()}")
    print(f"Validation set size: {val_mask.sum().item()}")
    print(f"Test set size: {test_mask.sum().item()}")

    net = FullNet(4, X.shape[1], data["num_classes"], num_heads=8)
    
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    print(f"Feature matrix shape: {X.shape}") 
    print(f"HG num_v: {HG.num_v}, Feature matrix shape[0]: {X.shape[0]}")  # Ensure both match

    HG = HG.to(X.device)
    net = net.to(device)
    
    # No early stopping
    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(400):
        # train
        train(net, X, HG, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, HG, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
