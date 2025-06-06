import torch
import ultralytics.nn.modules
import ultralytics.nn.tasks
from ultralytics import YOLO
import os 

base_dir = os.path.dirname(__file__)
yaml_file = os.path.join(base_dir, 'src/data.yaml')

model = YOLO('yolov8n.pt')

# Train on your custom door-window dataset
model.train(
    data=yaml_file,  
    epochs=100,
    imgsz=640,
    batch=16,
    name='door_window_yolov8',
    device='cpu'  
)
