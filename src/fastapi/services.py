import json
import requests


class BarcodeOCR:
    def __init__(self, config: dict):
        self.addr = config["addr"]

    def predict(self, image):
        return requests.put(self.addr, data=image)


class TorchServe:
    def __init__(self, config: dict):
        print(f"[TorchServe] [init] start {config=}")
        self.health_check_addr = config["health_check_addr"]
        print(f"[TorchServe] [init] end")
        

    def health_check(self):
        print(f"[TorchServe] [health_check] start")
        print(f"[TorchServe] [health_check] {self.health_check_addr=}")
        resp = requests.get(self.health_check_addr)
        if resp.ok:
            return json.loads(resp.text)
        else:
            return {"status": "Unavailable"}
        print(f"[TorchServe] [health_check] end {result=}")
        return result
