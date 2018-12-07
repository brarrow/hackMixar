import json


class Config:
    conditions = []

    def __init__(self, json):
        self.parsJSON(json)

    def parsJSON(self, json_data):
        d = json.loads(json_data)
        self.conditions = d.get("conditions")

        print(d)
        return d


Config = Config("{\"conditions\":[\
 {\"type\":0,\"operation\":2,\"value\":0,\"event\":0,\"area\":[90,50,183,166]}\
]}")
