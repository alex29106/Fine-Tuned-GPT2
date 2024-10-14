from pandas import read_parquet
import json

data = read_parquet("../data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
lis = []
for i in range(len(data)):
    lis.append({"text": data["instruction"][i] + " " + data["input"][i], "output": data["output"][i]})

with open("../data/Dataset1.json", "w") as f:
    json.dump(lis, f, indent=4)
    f.close()
