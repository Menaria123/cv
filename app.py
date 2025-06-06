from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os

# === CONFIGURATION ===
BASE_DIR       = os.getcwd()
IMAGES_DIR     = os.path.join(BASE_DIR, "Images")
TRAIN_DIR      = os.path.join(IMAGES_DIR, "train")
VAL_DIR        = os.path.join(IMAGES_DIR, "val")
DATA_YAML_PATH = os.path.join(BASE_DIR, "data.yaml")
MODEL_PATH     = os.path.join(BASE_DIR, "runs/detect/door_window_yolov8/weights/best.pt")

# === ENSURE FOLDERS EXIST ===
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# === WRITE data.yaml ===
yaml_content = f"""
train: {TRAIN_DIR}
val: {VAL_DIR}

nc: 2
names: ['door', 'window']
""".strip()

with open(DATA_YAML_PATH, "w") as f:
    f.write(yaml_content)

# === TRAIN YOLOv8 MODEL ===
print("🛠️  Starting training...")
model = YOLO("yolov8n.pt")
model.train(
    data=DATA_YAML_PATH,
    epochs=100,
    imgsz=640,
    batch=16,
    name="door_window_yolov8",
    device="cpu"
)
print("✅ Training complete.")

# === LOAD TRAINED MODEL FOR INFERENCE ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("📦 Loaded model with classes:", model.names)

# === FLASK APP ===
app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    results = model.predict(img, conf=0.05, imgsz=640)

    detections = []
    for r in results:
        for box in r.boxes:
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
