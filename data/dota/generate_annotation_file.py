import json
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

names = list(model.names.values())

with open("data/dota/obj_idx_to_labels.json", "w") as f:
    json.dump(names, f)
