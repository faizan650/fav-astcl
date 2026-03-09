import requests

class HereTrafficClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://traffic.ls.hereapi.com/traffic/6.3/flow.json"

    def get_flow(self, bbox):
        params = {
            "apiKey": self.api_key,
            "bbox": bbox,
            "responseattributes": "sh,fc"
        }
        r = requests.get(self.url, params=params)
        r.raise_for_status()
        return r.json()
