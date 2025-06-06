import torch
import ultralytics.nn.modules
import ultralytics.nn.tasks
from ultralytics import YOLO
import os 

base_dir = os.path.dirname(__file__)  # Path where this script is located
data_yaml = os.path.join(base_dir, "data/data.yaml")
# Load base model (YOLOv8n pretrained weights)
model = YOLO('yolov8n.pt')

# Train on your custom door-window dataset
model.train(
    data=data_yaml,  
    epochs=100,
    imgsz=640,
    batch=16,
    name='door_window_yolov8',
    device='cpu'  
)
