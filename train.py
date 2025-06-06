import torch
import ultralytics.nn.modules
import ultralytics.nn.tasks
from ultralytics import YOLO
import os 


model = YOLO('yolov8n.pt')

# Train on your custom door-window dataset
model.train(
    data="src/mydata/data.yaml",  
    epochs=100,
    imgsz=640,
    batch=16,
    name='door_window_yolov8',
    device='cpu'  
)
