import torch
import ultralytics.nn.tasks

# Allowlist the DetectionModel class to bypass the restricted loading error
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='door_window_yolov8',
    device='0'
)
