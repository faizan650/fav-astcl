import datetime
import numpy as np

INDIAN_HOLIDAYS = {
    "2024-10-02",
    "2024-11-01"
}

def calendar_features(ts=None):
    if ts is None:
        ts = datetime.datetime.now()

    hour = ts.hour
    weekday = ts.weekday()

    return [
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        int(weekday >= 5),
        int(ts.strftime("%Y-%m-%d") in INDIAN_HOLIDAYS)
    ]
