import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pickle
import datetime
import requests
import time

from model import FAV_ASTCL

# =====================================================
# 🔑 CONFIG
# =====================================================

TOMTOM_API_KEY = "xxxxxx"
WEATHER_API_KEY = "xxxxx"

SIMULATE = True   # ⚠ Set True if TomTom 403 error

TRAIN_MEAN = 50.7  # Replace with your real training mean
TRAIN_STD  = 20.2   # Replace with your real training std

CITY = "Hyderabad"

MODEL_PATH = "results/fav_astcl.pth"
ADJ_PATH = "datasets/METR-LA/adj_mx.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Define Hyderabad Sensors
# =====================================================

SENSORS = {
    "HYD_01": (17.4239, 78.4484),  # Punjagutta
    "HYD_02": (17.4435, 78.3772),  # Hitech City
    "HYD_03": (17.4399, 78.4983),  # Secunderabad
    "HYD_04": (17.4401, 78.3489),  # Gachibowli
    "HYD_05": (17.3450, 78.5480),  # LB Nagar
}

# =====================================================
# Load adjacency
# =====================================================

with open(ADJ_PATH, "rb") as f:
    adj_data = pickle.load(f)

adj = adj_data[0] if isinstance(adj_data, tuple) else adj_data
adj = torch.tensor(adj, dtype=torch.float32)

# =====================================================
# Load trained model
# =====================================================

model = FAV_ASTCL(adj).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("🚀 Multi-Sensor Live Streaming Started")

# =====================================================
# STREAMING LOOP
# =====================================================

while True:

    print("\n=================================================")
    print("🕒 Time:", datetime.datetime.now())
    print("=================================================")

    # -------------------------------------------------
    # Fetch Weather (once per cycle)
    # -------------------------------------------------

    try:
        weather_url = "http://api.weatherapi.com/v1/current.json"
        weather_params = {
            "key": WEATHER_API_KEY,
            "q": CITY
        }

        w = requests.get(weather_url, params=weather_params)
        w.raise_for_status()
        weather_json = w.json()

        temp = weather_json["current"]["temp_c"]
        humidity = weather_json["current"]["humidity"]
        wind = weather_json["current"]["wind_kph"]

    except:
        temp, humidity, wind = 30, 50, 10

    hour = datetime.datetime.now().hour
    weekday = datetime.datetime.now().weekday()
    is_weekend = 1 if weekday >= 5 else 0

    # -------------------------------------------------
    # Process Each Sensor
    # -------------------------------------------------

    for sensor_id, (lat, lon) in SENSORS.items():

        # ---------------------------
        # Traffic Fetch
        # ---------------------------

        if SIMULATE:
            current_speed = np.random.randint(15, 60)
            free_speed = 60
        else:
            try:
                traffic_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
                params = {
                    "key": TOMTOM_API_KEY,
                    "point": f"{lat},{lon}"
                }

                r = requests.get(traffic_url, params=params)
                r.raise_for_status()
                data = r.json()

                segment = data["flowSegmentData"]
                current_speed = segment["currentSpeed"]
                free_speed = segment["freeFlowSpeed"]

            except:
                current_speed = 30.0
                free_speed = 45.0

        # ---------------------------
        # Congestion Detection
        # ---------------------------

        congestion_ratio = current_speed / (free_speed + 1e-5)
        status = "🚨 CONGESTED" if congestion_ratio < 0.6 else "🟢 NORMAL"

        print(f"\n📍 Sensor: {sensor_id}")
        print("   Current Speed:", current_speed, "km/h")
        print("   Status:", status)

        # ---------------------------
        # Normalize Input
        # ---------------------------

        normalized_speed = (current_speed - TRAIN_MEAN) / TRAIN_STD

        traffic_vector = np.ones(207) * normalized_speed
        traffic_window = np.tile(traffic_vector, (12, 1))
        traffic_window = traffic_window.reshape(1, 12, 207, 1)

        exo = np.array(
            [temp, humidity, wind, hour, weekday, is_weekend],
            dtype=np.float32
        )

        exo_window = np.tile(exo, (12, 1)).reshape(1, 12, 6)

        x = torch.tensor(traffic_window, dtype=torch.float32).to(DEVICE)
        exo_tensor = torch.tensor(exo_window, dtype=torch.float32).to(DEVICE)

        # ---------------------------
        # Forecast
        # ---------------------------

        with torch.no_grad():
            prediction = model(x, exo_tensor)

        prediction = prediction.cpu().numpy()
        prediction_raw = prediction * TRAIN_STD + TRAIN_MEAN

        print("   📈 Forecast:")
        print("      05 min:", round(float(prediction_raw[0, 0, 0]), 2), "km/h")
        print("      10 min:", round(float(prediction_raw[0, 0, 1]), 2), "km/h")
        print("      15 min:", round(float(prediction_raw[0, 0, 2]), 2), "km/h")

    print("\n🔄 Updating in 60 seconds...")
    time.sleep(60)
