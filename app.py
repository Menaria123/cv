from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
import argparse

app = Flask(__name__)
MODEL_PATH = "runs/detect/door_window_yolov8/weights/best.pt"

def train_model():
    print("🛠️ Starting training...")
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='door_window_yolov8',
        device='cpu'
    )
    print("✅ Training complete.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")
    print("📦 Loading trained model...")
    return YOLO(MODEL_PATH)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    results = app.model.predict(image, conf=0.05, imgsz=640)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = app.model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            })

    return jsonify({"detections": detections})

def start_server():
    app.model = load_model()
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 Trainer and Inference API")
    parser.add_argument('--train', action='store_true', help="Train the YOLOv8 model")
    args = parser.parse_args()

    if args.train:
        train_model()
    else:
        start_server()
