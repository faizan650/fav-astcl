import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import numpy as np
import pickle
import datetime

from model import FAV_ASTCL

# =====================================================
# CONFIG
# =====================================================

SIMULATE = True

TRAIN_MEAN = 34.5
TRAIN_STD  = 12.8

CITY = "Hyderabad"

# =====================================================
# SENSOR DEFINITIONS
# =====================================================

SENSORS = {
    "HYD_01": (17.4239, 78.4484),
    "HYD_02": (17.4435, 78.3772),
    "HYD_03": (17.4399, 78.4983),
    "HYD_04": (17.4401, 78.3489),
    "HYD_05": (17.3450, 78.5480),
}

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    with open("datasets/METR-LA/adj_mx.pkl", "rb") as f:
        adj_data = pickle.load(f)

    adj = adj_data[0] if isinstance(adj_data, tuple) else adj_data
    adj = torch.tensor(adj, dtype=torch.float32)

    model = FAV_ASTCL(adj)
    model.load_state_dict(torch.load("results/fav_astcl.pth", map_location="cpu"))
    model.eval()

    return model

model = load_model()

# =====================================================
# PAGE SETUP
# =====================================================

st.set_page_config(page_title="Traffic Dashboard", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚦 Smart Traffic Dashboard")
st.subheader("FAV-ASTCL Real-Time Forecasting")

# =====================================================
# AUTO REFRESH (1 minute)
# =====================================================

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=60000)

# =====================================================
# TIME DISPLAY
# =====================================================

st.markdown(f"🕒 **Time:** `{datetime.datetime.now()}`")
st.markdown("---")

# =====================================================
# WEATHER (SIMULATED)
# =====================================================

temp, humidity, wind = 30, 50, 10

hour = datetime.datetime.now().hour
weekday = datetime.datetime.now().weekday()
is_weekend = 1 if weekday >= 5 else 0

# =====================================================
# SENSOR GRID
# =====================================================

cols = st.columns(3)

for i, (sensor_id, (lat, lon)) in enumerate(SENSORS.items()):

    with cols[i % 3]:

        # ---------------------------
        # Simulated Traffic
        # ---------------------------
        current_speed = np.random.randint(15, 60)
        free_speed = 60

        ratio = current_speed / free_speed

        # ---------------------------
        # Status & Color
        # ---------------------------
        if ratio < 0.6:
            status = "🚨 CONGESTED"
            color = "#ef4444"
        else:
            status = "🟢 NORMAL"
            color = "#22c55e"

        # ---------------------------
        # Normalize
        # ---------------------------
        norm_speed = (current_speed - TRAIN_MEAN) / TRAIN_STD

        traffic_vector = np.ones(207) * norm_speed
        traffic_window = np.tile(traffic_vector, (12, 1)).reshape(1, 12, 207, 1)

        exo = np.array(
            [temp, humidity, wind, hour, weekday, is_weekend],
            dtype=np.float32
        )

        exo_window = np.tile(exo, (12, 1)).reshape(1, 12, 6)

        x = torch.tensor(traffic_window, dtype=torch.float32)
        exo_tensor = torch.tensor(exo_window, dtype=torch.float32)

        # ---------------------------
        # Prediction
        # ---------------------------
        with torch.no_grad():
            pred = model(x, exo_tensor).numpy()

        pred_raw = pred * TRAIN_STD + TRAIN_MEAN

        p15 = round(float(pred_raw[0,0,0]), 2)
        p30 = round(float(pred_raw[0,0,1]), 2)
        p60 = round(float(pred_raw[0,0,2]), 2)

        # ---------------------------
        # CARD UI (ALL IN ONE)
        # ---------------------------
        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #1e293b, #0f172a);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid {color};
            box-shadow: 0px 0px 15px {color};
            margin-bottom: 20px;
        ">

            <h3 style="color:{color};">📍 {sensor_id}</h3>

            <p style="font-size:18px; color:{color};">
                <b>Speed:</b> {current_speed} km/h
            </p>

            <p style="font-size:18px; color:{color};">
                <b>Status:</b> {status}
            </p>

            <hr style="border-color:{color};">

            <p style="font-size:18px; color:{color};"><b>Forecast:</b></p>

            <ul style="color:{color}; font-size:16px;">
                <li>15 min: {p15} km/h</li>
                <li>30 min: {p30} km/h</li>
                <li>60 min: {p60} km/h</li>
            </ul>

        </div>
        """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("🚀 FAV-ASTCL | Intelligent Traffic Forecasting System")