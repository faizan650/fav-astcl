# FAV-ASTCL: Technical Documentation
**Forecasting Aware Versatile Adaptive Spatio-Temporal Context Learning**

This documentation is a practical guide to the FAV-ASTCL framework. Our goal with this project was to tackle the unpredictable "volatility" in real-world traffic data by making the spatio-temporal graph learning more **aware** and **versatile**.

---

## 🎨 Algorithm Conceptualization

The core idea is that traffic is a "contextual" problem. You can't just look at past speeds; you have to understand the environment (weather, time, city-layout) to know if a slowdown is a normal rush hour or an emergency event.

### 1. Dynamics over Distance (The Graph Problem)
Traditional maps use physical distance. We implemented a **Learnable Context Selector**. 
- It uses a blend of a static map ($A_{static}$) sssand a generated similarity matrix ($A_{dyn}$).
- The model chooses how much to trust each one using a learnable $\alpha$ parameter. 
- During a jam at **Hitech City**, the model effectively "re-routes" its attention by updating this matrix every 5 minutes.

### 2. The Context Gating (Weather & Time)
We found that just adding weather data often "noises up" the model. 
- We built a `Gating Mechanism` that acts like a pressure valve.
- If the current traffic window (the "actual" speed) doesn't correlate with a weather change (the "context"), the model suppresses the weather signal. This prevents the model from "panicking" when it's raining but traffic is still moving fine.

### 3. Residual Online Adapter
This was a key fix during development. Standard GRUs tend to "lag" when a sudden catastrophe happens. We added a small **Residual MLP** that specifically looks at the prediction error and corrects it. It's essentially a "high-resolution" layer that catches what the main graph layers might overlook.

---

## 🌎 Live Environment Data Flow

The project is designed for **real-time serving**:

1.  **Ingestion**: Every 15 minutes, we hit the **TomTom Traffic API** and **WeatherAPI**.
2.  **Windowing**: We stack the last 12 time-steps (60 mins) into a tensor.
3.  **Forward Pass**: The FAV-ASTCL model processes this "Spatiotemporal Cube" alongside the gated weather context.
4.  **Prediction**: We output a forecast for the next 15 minutes (split into 5, 10, 15 min horizons).
5.  **Serving**: A Flask-based UI vizualizes this for 5 key hubs: Punjagutta, Hitech City, Secunderabad, Gachibowli, and LB Nagar.

---

## 📊 Deployment Performance (Actual Observed Metrics)

The following benchmark demonstrates the leap in stability we achieved:

| Model | MAE | RMSE | MAPE | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| ASTCL (Baseline) | 9.034 | 16.173 | 23.47% | 76.53% |
| **FAV-ASTCL (Ours)** | **3.633** | **8.842** | **8.22%** | **91.78%** |

---

## 🔧 File Map
- `src/model.py`: Core logic for the Graph Selection and Gating.
- `src/live/server.py`: Flask dashboard and live API integration.
- `src/train_replay.py`: The training loop we used to reach the 91% accuracy mark.
- `datasets/exo_hyderabad.npy`: Our curated exogenous feature store for the city.
