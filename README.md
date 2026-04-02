# FAV-ASTCL: Forecasting Aware Versatile Adaptive Spatio-Temporal Context Learning

FAV-ASTCL is a practical deep learning framework developed to tackle the "messy" reality of urban traffic. Unlike standard models that work well on clean historical data but fail during sudden storms or accidents, this project implements a **Versatile Adaptive** approach to keep predictions stable when things get volatile.

## What's actually inside?
Most traffic models use a fixed map. We found that doesn't work for a city like Hyderabad where a single rain shower in Hitech City changes the entire network's "connectivity" in minutes. 

Our solution focuses on:
- **Dynamic Context Selection**: The model literally "re-wires" its spatial understanding every few minutes based on current flow.
- **Exogenous Gating**: A mechanism that filters out "weather noise" so the model doesn't overreact to every cloud.
- **Online Adaptation**: A residual layer that catches the small spikes that standard GRUs often miss.

## Quick Data Comparison
| System | MAE (Error) | Accuracy |
| :--- | :--- | :--- |
| ASTCL (Baseline) | 9.034 | 76.5% |
| **FAV-ASTCL** | **3.633** | **91.8%** |

## Getting it Running

### 1. Data Prep
We used METR-LA for the baseline, but the real test is our Hyderabad live set. You'll need to drop your `.npy` or `.h5` files into the `datasets/` folder.

### 2. Live Dashboard
If you want to see the model running against real TomTom and WeatherAPI data:
1. Open `src/live/server.py`.
2. Add your own `WEATHER_API_KEY` (I've left a placeholder, but it might hit limits).
3. Run the server:
   ```bash
   python src/live/server.py
   ```
4. Visit `localhost:5000` to see the live predictions across the 5 Hyderabad hubs.

### 3. Training from Scratch
To retrain on your own data:
```bash
python src/train_replay.py
```

## Structure
- `src/model.py`: The actual FAV-ASTCL architecture (Selector, Gating, Adapter).
- `src/live/`: The Flask & JS bits for the real-time visualization.
- `datasets/`: Where the traffic and weather "context" lives.
