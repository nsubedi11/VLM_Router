import json
import os

sample_path = "data/processed/sample_10.json"

with open(sample_path, "r") as f:
    data = json.load(f)

print("num samples:", len(data))
print()

item = data[0]
print("id:", item["id"])
print("num videos:", len(item["video"]))
print("prompt:")
print(item["conversations"][0]["value"])
print()
print("answer:", item["conversations"][1]["value"])
print()

for i, vp in enumerate(item["video"], 1):
    print(f"video {i}: {vp} | exists={os.path.exists(vp)}")