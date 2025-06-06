from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
import torch
import ultralytics.nn.modules
import ultralytics.nn.tasks
import torch
# Load base model (YOLOv8n pretrained weights)
model = YOLO('yolov8n.pt')

# Train on your custom door-window dataset
model.train(
    data=r"data.yaml",  
    epochs=100,
    imgsz=640,
    batch=16,
    name='door_window_yolov8',
    device='cpu'  
)
app = Flask(__name__)


MODEL_PATH = r".\door and window detection\runs\detect\door_window_yolov813\weights\best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)


print("Loaded model with classes:", model.names)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    results = model.predict(image, conf=0.05, imgsz=640)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            })

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(debug=True)
