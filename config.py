import json


class Config:
    # area is x, y, w, h
    conditions = []

    def __init__(self, json):
        self.parsJSON(json)

    def parsJSON(self, json_data):
        d = json.loads(json_data)
        self.conditions = d.get("conditions")
        for i in range(len(self.conditions)):
            area = self.conditions[i]["area"]
            area[2] -= area[0]
            area[3] -= area[2]
            self.conditions[i]["area"] = area
        print(d)
        return d


Config = Config("{\"conditions\":[\
 {\"type\":0,\"operation\":2,\"value\":0,\"event\":0,\"area\":[90,50,183,166]}\
]}")
