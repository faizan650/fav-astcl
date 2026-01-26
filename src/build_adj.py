# The spatial adjacency matrix was constructed using geographical distances between sensors based on their latitude and longitude,
# following a Gaussian kernel formulation,
# which is consistent with prior spatio-temporal traffic forecasting studies

import pandas as pd
import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt

# ----------------------------
# Haversine distance
# ----------------------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # km

# ----------------------------
# Load sensor locations
# ----------------------------
df = pd.read_csv("datasets/PEMS-BAY/graph_sensor_locations.csv")
sensor_ids = df["sensor_id"].astype(str).tolist()
coords = df[["longitude", "latitude"]].values

N = len(sensor_ids)
dist_matrix = np.zeros((N, N))

# ----------------------------
# Compute distance matrix
# ----------------------------
for i in range(N):
    for j in range(N):
        dist_matrix[i, j] = haversine(
            coords[i][0], coords[i][1],
            coords[j][0], coords[j][1]
        )

# ----------------------------
# Convert distance → adjacency
# ----------------------------
sigma = np.std(dist_matrix)
adj = np.exp(-(dist_matrix ** 2) / (sigma ** 2))

# remove self-loops if desired
np.fill_diagonal(adj, 0)

# ----------------------------
# Save adjacency (ASTCL-compatible)
# ----------------------------
with open("datasets/PEMS-BAY/adj_mx.pkl", "wb") as f:
    pickle.dump((adj, sensor_ids), f)

print("Adjacency matrix created:", adj.shape)
