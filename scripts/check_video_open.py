import json
import cv2

with open("data/processed/sample_10.json", "r") as f:
    data = json.load(f)

video_path = data[0]["video"][0]
print("testing:", video_path)

cap = cv2.VideoCapture(video_path)
ok, frame = cap.read()
cap.release()

print("opened:", ok)
if ok and frame is not None:
    print("frame shape:", frame.shape)
else:
    print("failed to read first frame")