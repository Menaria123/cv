from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)

# ✅ Load your trained model (adjust this path if needed)
MODEL_PATH = r"E:\D\door and window detection\runs\detect\door_window_yolov8\weights\best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Predict with model
    results = model.predict(image, conf=0.25, imgsz=640)

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
