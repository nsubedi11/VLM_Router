import json

with open("data/processed/all_6tasks.json") as f:
    data = json.load(f)

all_videos = []
for item in data:
    if "videos" in item:
        all_videos.extend(item["videos"])
    elif "video" in item:
        if isinstance(item["video"], list):
            all_videos.extend(item["video"])
        else:
            all_videos.append(item["video"])

print("Total referenced videos:", len(all_videos))
print("Unique referenced videos:", len(set(all_videos)))
