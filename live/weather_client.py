import requests

class WeatherClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_weather(self, city):
        url = "http://api.weatherapi.com/v1/current.json"
        params = {"key": self.api_key, "q": city}
        r = requests.get(url, params=params)
        r.raise_for_status()
        d = r.json()["current"]

        return [
            d["temp_c"],
            d["precip_mm"],
            d["humidity"],
            d["vis_km"]
        ]
