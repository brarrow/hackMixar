import json


class Config:
    # area is x, y, w, h, isTrue (boolean, esli vipolnyaetsya)
    conditions = []

    def __init__(self, json):
        self.parsJSON(json)

    def get_cond(self, cond_type):
        res = []
        for el in self.conditions:
            if el["type"] == cond_type:
                res.append(el)
        return res


    def parsJSON(self, json_data):
        d = json.loads(json_data)
        self.conditions = d.get("conditions")
        for i in range(len(self.conditions)):
            area = self.conditions[i]["area"]
            area[2] -= area[0]
            area[3] -= area[2]
            self.conditions[i]["area"] = area
            self.conditions[i]["isTrue"] = False
        print(d)
        return d


config = Config("{\"conditions\":[\
 {\"type\":0,\"operation\":2,\"value\":0,\"event\":0,\"area\":[90,50,183,166]}\
]}")
