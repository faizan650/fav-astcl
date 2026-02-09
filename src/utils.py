import pickle

def load_adj(path):
    with open(path, "rb") as f:
        adj, _, _ = pickle.load(f)
    return adj
