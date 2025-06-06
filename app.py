from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os

# === CONFIGURATION ===
MODEL_PATH = "runs/detect/door_window_yolov8/weights/best.pt"
DATA_YAML = "data.yaml"

# === TRAINING FUNCTION ===
def train_model():
    print("🛠️ Starting training...")
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        name='door_window_yolov8',
        device='cpu'
    )
    print("✅ Training complete.")

# === CHECK FOR EXISTING MODEL OR TRAIN ===
if not os.path.exists(MODEL_PATH):
    train_model()

# === LOAD TRAINED MODEL ===
model = YOLO(MODEL_PATH)
print("Loaded model with classes:", model.names)

# === FLASK APP ===
app = Flask(__name__)

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
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var if set
    app.run(host='0.0.0.0', port=port, debug=True)
