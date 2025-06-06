import os
import random
import shutil
import threading
import time

from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────
#                 CONFIGURATION
# ─────────────────────────────────────────────────────────────

BASE_DIR       = os.getcwd()                              # e.g. /opt/render/project/src
IMAGES_DIR     = os.path.join(BASE_DIR, "Images")         # /opt/render/project/src/Images
TRAIN_DIR      = os.path.join(IMAGES_DIR, "train")        # /opt/render/project/src/Images/train
VAL_DIR        = os.path.join(IMAGES_DIR, "val")          # /opt/render/project/src/Images/val
DATA_YAML_PATH = os.path.join(BASE_DIR, "data.yaml")
WEIGHTS_DIR    = os.path.join(BASE_DIR, "runs/detect/door_window_yolov8/weights")
BEST_WEIGHTS   = os.path.join(WEIGHTS_DIR, "best.pt")
YOLO_BASE      = "yolov8n.pt"   # pretrained backbone to download

# Supported image & label extensions (YOLO expects .txt alongside each image)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
LABEL_EXTS = {".txt"}           # YOLO‐format labels

TRAIN_RATIO = 0.8  # 80% train / 20% val split if done from Images/

# ─────────────────────────────────────────────────────────────
#          DATA PREPARATION & TRAINING (background thread)
# ─────────────────────────────────────────────────────────────

def prepare_and_train():
    """
    1) Ensure Images/train & Images/val folders exist.
    2) If both are empty but Images/ has images, split & move them.
    3) Write a minimal data.yaml that points to these two folders.
    4) Call YOLOv8.train(...) to produce best.pt (if best.pt does not already exist).
    """
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR,   exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # If both train/ and val/ are empty, but Images/ has images, do the 80/20 split
    if not os.listdir(TRAIN_DIR) and not os.listdir(VAL_DIR):
        # 1a) Gather all image files directly under Images/
        all_images = []
        for fname in sorted(os.listdir(IMAGES_DIR)):
            fpath = os.path.join(IMAGES_DIR, fname)
            if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                all_images.append(fname)

        if all_images:
            # 1b) Pair each image with its same‐basename .txt (if exists)
            pairs = []
            for img_fname in all_images:
                basename, _ = os.path.splitext(img_fname)
                label_fname = None
                for ext in LABEL_EXTS:
                    candidate = f"{basename}{ext}"
                    if os.path.isfile(os.path.join(IMAGES_DIR, candidate)):
                        label_fname = candidate
                        break
                pairs.append((img_fname, label_fname))

            print(f"[dataset] Found {len(pairs)} image(s) under Images/ (with {sum(1 for _,lbl in pairs if lbl)} labels).")
            random.shuffle(pairs)
            n_train = int(len(pairs) * TRAIN_RATIO)
            train_pairs = pairs[:n_train]
            val_pairs   = pairs[n_train:]

            print(f"[dataset] → {len(train_pairs)} pairs → TRAIN/")
            print(f"[dataset] → {len(val_pairs)} pairs → VAL/")

            def move_pair(dst_dir, img_fname, lbl_fname):
                src_img = os.path.join(IMAGES_DIR, img_fname)
                dst_img = os.path.join(dst_dir, img_fname)
                shutil.move(src_img, dst_img)
                if lbl_fname:
                    src_lbl = os.path.join(IMAGES_DIR, lbl_fname)
                    dst_lbl = os.path.join(dst_dir, lbl_fname)
                    shutil.move(src_lbl, dst_lbl)
                else:
                    print(f"[dataset] ⚠️  No label for {img_fname}; moved image only.")

            for img_fname, lbl_fname in train_pairs:
                move_pair(TRAIN_DIR, img_fname, lbl_fname)
            for img_fname, lbl_fname in val_pairs:
                move_pair(VAL_DIR, img_fname, lbl_fname)

            print("[dataset] ✅ “train/val” split done.")
        else:
            print("[dataset] ℹ️  No images found under Images/. Skipping split.")
    else:
        print("[dataset] ℹ️  train/val folders not empty; skipping dataset split.")

    # 2) Write data.yaml
    yaml_txt = f"""
train: {TRAIN_DIR}
val:   {VAL_DIR}

nc: 2
names: ['door', 'window']
""".strip()

    with open(DATA_YAML_PATH, "w") as f:
        f.write(yaml_txt)
    print(f"[dataset] ✅ Wrote data.yaml:\n{yaml_txt}\n")

    # 3) If best.pt already exists, skip training
    if os.path.exists(BEST_WEIGHTS):
        print(f"[train] 📦 Found existing {BEST_WEIGHTS}; skipping training.")
        return

    # 4) Launch YOLOv8 training (this can take minutes/hours)
    print("[train] 🛠️  Starting YOLOv8 training (background thread)...")
    model = YOLO(YOLO_BASE)
    model.train(
        data=DATA_YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        name="door_window_yolov8",
        device="cpu"
    )
    print("[train] ✅ Training finished. best.pt should now exist.")


# Start the training in a daemon thread
train_thread = threading.Thread(target=prepare_and_train, daemon=True)
train_thread.start()

# ─────────────────────────────────────────────────────────────
#                  FLASK APP FOR INFERENCE
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
_model = None  # Will hold the loaded YOLO model once best.pt exists

def try_load_model():
    """
    If best.pt exists and we haven’t loaded it yet, load it into _model.
    """
    global _model
    if _model is None and os.path.isfile(BEST_WEIGHTS):
        print(f"[inference] 📦 Loading model from {BEST_WEIGHTS}")
        _model = YOLO(BEST_WEIGHTS)
        print(f"[inference] 📦 Model loaded; classes = {_model.names}")

def model_loader_loop():
    """
    Every 10 seconds, check for BEST_WEIGHTS and load if present.
    """
    while True:
        try_load_model()
        time.sleep(10)

# Start the model‐loader thread
loader_thread = threading.Thread(target=model_loader_loop, daemon=True)
loader_thread.start()


@app.route("/detect", methods=["POST"])
def detect():
    """
    1. If _model is not yet loaded (best.pt missing), return 503 (retry later).
    2. Otherwise, run model.predict() on the uploaded image and return detections JSON.
    """
    if _model is None:
        return jsonify({"error": "Model not yet trained/loaded. Please retry in a minute."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    results = _model.predict(img, conf=0.05, imgsz=640)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = _model.names[cls_id]

            detections.append({
                "label":      label,
                "confidence": round(conf, 2),
                "bbox":       [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            })

    return jsonify({"detections": detections})


# ─────────────────────────────────────────────────────────────
#                        STARTUP
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Bind to 0.0.0.0 and use Render’s $PORT (or default 5000 locally)
    port = int(os.environ.get("PORT", 5000))
    print(f"[server] 🚀 Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
