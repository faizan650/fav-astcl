import pickle

def load_adj(path):
    with open(path, "rb") as f:
        try:
            adj_data = pickle.load(f)
        except UnicodeDecodeError:
            # Python2 → Python3 compatibility
            adj_data = pickle.load(f, encoding="latin1")

    # adj_mx.pkl may be tuple or ndarray
    if isinstance(adj_data, tuple):
        adj = adj_data[0]
    else:
        adj = adj_data

    return adj
