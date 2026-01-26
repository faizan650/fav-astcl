import pandas as pd
import numpy as np

# ----------------------------
# Paths
# ----------------------------
TRAFFIC_CSV = "datasets/PEMS-BAY/PEMS-BAY.csv"
SENSOR_LOCATIONS = "datasets/PEMS-BAY/graph_sensor_locations_bay.csv"
OUT_PATH = "datasets/PEMS-BAY/data-BAY.npz"

# ----------------------------
# Load sensor locations
# ----------------------------
print("Loading sensor locations...")
sensor_df = pd.read_csv(SENSOR_LOCATIONS)

# Ensure sensor_id is string (important for matching columns)
sensor_df["sensor_id"] = sensor_df["sensor_id"].astype(str)

# Sensor order (this becomes node order)
sensor_ids = sensor_df["sensor_id"].tolist()
print(f"Number of sensors: {len(sensor_ids)}")

# ----------------------------
# Load traffic data
# ----------------------------
print("Loading traffic CSV...")
traffic_df = pd.read_csv(TRAFFIC_CSV)

# First column is timestamp
timestamp_col = traffic_df.columns[0]

# Convert column names to string for matching
traffic_df.columns = traffic_df.columns.astype(str)

# ----------------------------
# Reorder traffic columns to match sensor_locations
# ----------------------------
missing = [sid for sid in sensor_ids if sid not in traffic_df.columns]
if len(missing) > 0:
    raise ValueError(f"Missing sensors in traffic data: {missing[:5]} ...")

print("Reordering traffic data to match sensor order...")
traffic_values = traffic_df[sensor_ids].values

# ----------------------------
# Handle missing values (simple forward fill)
# ----------------------------
traffic_values = pd.DataFrame(traffic_values).fillna(method="ffill").fillna(method="bfill").values

# ----------------------------
# Final shape: (T, N, 1)
# ----------------------------
data = np.expand_dims(traffic_values, axis=-1)

print("Final data shape:", data.shape)

# ----------------------------
# Save to NPZ
# ----------------------------
np.savez_compressed(OUT_PATH, data=data)
print(f"Saved data.npz to {OUT_PATH}")
