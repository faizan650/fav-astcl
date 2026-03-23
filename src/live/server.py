import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
import pickle
import datetime
import requests

from model import FAV_ASTCL

# =====================================================
# CONFIG
# =====================================================

SIMULATE = True

TRAIN_MEAN = 34.5
TRAIN_STD  = 12.8

CITY = "Hyderabad"
WEATHER_API_KEY = "0be9c9c8dc7b4414af960654262502"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "results", "fav_astcl.pth")
ADJ_PATH   = os.path.join(BASE_DIR, "datasets", "METR-LA", "adj_mx.pkl")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# SENSORS
# =====================================================

SENSORS = {
    "HYD_01": {"name": "Punjagutta",    "lat": 17.4239, "lon": 78.4484},
    "HYD_02": {"name": "Hitech City",   "lat": 17.4435, "lon": 78.3772},
    "HYD_03": {"name": "Secunderabad",  "lat": 17.4399, "lon": 78.4983},
    "HYD_04": {"name": "Gachibowli",    "lat": 17.4401, "lon": 78.3489},
    "HYD_05": {"name": "LB Nagar",      "lat": 17.3450, "lon": 78.5480},
}

# =====================================================
# LOAD MODEL (once at startup)
# =====================================================

with open(ADJ_PATH, "rb") as f:
    adj_data = pickle.load(f)

adj = adj_data[0] if isinstance(adj_data, tuple) else adj_data
adj = torch.tensor(adj, dtype=torch.float32)

model = FAV_ASTCL(adj).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(f"✅ Model loaded on {DEVICE}")

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)


def fetch_weather():
    """Fetch live weather or return simulated defaults."""
    try:
        if not SIMULATE:
            url = "http://api.weatherapi.com/v1/current.json"
            params = {"key": WEATHER_API_KEY, "q": CITY}
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            d = r.json()["current"]
            return {
                "temperature": d["temp_c"],
                "humidity": d["humidity"],
                "wind_speed": d["wind_kph"],
            }
    except Exception:
        pass

    return {"temperature": 30.0, "humidity": 50.0, "wind_speed": 10.0}


def fetch_traffic(lat, lon):
    """Fetch live traffic or return simulated defaults."""
    if SIMULATE:
        current_speed = float(np.random.randint(15, 65))
        free_speed = 60.0
    else:
        try:
            url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            params = {"key": "XgRR1pB0yE18lITkmlfVJgmtBZX3EIj2", "point": f"{lat},{lon}"}
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            seg = r.json()["flowSegmentData"]
            current_speed = float(seg["currentSpeed"])
            free_speed = float(seg["freeFlowSpeed"])
        except Exception:
            current_speed = 30.0
            free_speed = 45.0

    return current_speed, free_speed


def run_inference(current_speed, weather, hour, weekday, is_weekend):
    """Run FAV-ASTCL inference and return denormalized predictions."""
    norm_speed = (current_speed - TRAIN_MEAN) / TRAIN_STD
    traffic_window = np.ones((1, 12, 207, 1), dtype=np.float32) * norm_speed

    exo = np.array(
        [weather["temperature"], weather["humidity"], weather["wind_speed"],
         hour, weekday, is_weekend],
        dtype=np.float32,
    )
    exo_window = np.tile(exo, (12, 1)).reshape(1, 12, 6)

    x = torch.tensor(traffic_window, dtype=torch.float32).to(DEVICE)
    exo_tensor = torch.tensor(exo_window, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred = model(x, exo_tensor).cpu().numpy()

    pred_raw = pred * TRAIN_STD + TRAIN_MEAN
    return {
        "5_min":  round(float(pred_raw[0, 0, 0]), 2),
        "10_min": round(float(pred_raw[0, 0, 1]), 2),
        "15_min": round(float(pred_raw[0, 0, 2]), 2),
    }


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.route("/api/predictions")
def predictions():
    now = datetime.datetime.now()
    hour = now.hour
    weekday = now.weekday()
    is_weekend = 1 if weekday >= 5 else 0

    weather = fetch_weather()

    sensors = []
    for sensor_id, info in SENSORS.items():
        current_speed, free_speed = fetch_traffic(info["lat"], info["lon"])

        congestion_ratio = round(current_speed / (free_speed + 1e-5), 3)
        congestion_status = "CONGESTED" if congestion_ratio < 0.6 else "NORMAL"

        forecast = run_inference(current_speed, weather, hour, weekday, is_weekend)

        sensors.append({
            "sensor_id": sensor_id,
            "name": info["name"],
            "latitude": info["lat"],
            "longitude": info["lon"],
            "current_speed": round(current_speed, 2),
            "free_flow_speed": round(free_speed, 2),
            "congestion_ratio": congestion_ratio,
            "congestion_status": congestion_status,
            "forecast": forecast,
        })

    return jsonify({
        "timestamp": now.isoformat(),
        "weather": weather,
        "time_features": {
            "hour": hour,
            "weekday": weekday,
            "weekday_name": now.strftime("%A"),
            "is_weekend": bool(is_weekend),
        },
        "sensors": sensors,
    })


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("🚀 Starting FAV-ASTCL Dashboard Server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
