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
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

BASE_DIR       = os.getcwd()                         # e.g. /opt/render/project/src
IMAGES_DIR     = os.path.join(BASE_DIR, "Images")     #   /opt/render/project/src/Images
TRAIN_DIR      = os.path.join(IMAGES_DIR, "train")    #   /opt/render/project/src/Images/train
VAL_DIR        = os.path.join(IMAGES_DIR, "val")      #   /opt/render/project/src/Images/val
DATA_YAML_PATH = os.path.join(BASE_DIR, "data.yaml")
WEIGHTS_DIR    = os.path.join(BASE_DIR, "runs/detect/door_window_yolov8/weights")
BEST_WEIGHTS   = os.path.join(WEIGHTS_DIR, "best.pt")
YOLO_BASE      = "yolov8n.pt"  # pretrained yolov8n

# If you have fewer/no labels yet, YOLO will treat unlabeled images as “no objects.”
# But ideally, each image X.png → X.txt in the same folder for YOLO format.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXTS = {".txt"}  # assume YOLO‐format annotations

TRAIN_RATIO = 0.8  # 80% train, 20% val if splitting

# ─────────────────────────────────────────────────────────────
# DATA PREPARATION & TRAINING (in a background thread)
# ─────────────────────────────────────────────────────────────

def prepare_and_train():
    """
    1. Ensure Images/train & Images/val exist.
    2. If train/val are empty but Images/ holds PNG/JPG files, split & move them.
    3. Write data.yaml.
    4. Call YOLOv8 .train(...) to produce best.pt.
    """
    # 1) Ensure folders
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # 2) Check if TRAIN_DIR is empty but Images/ has images
    existing_train = os.listdir(TRAIN_DIR)
    existing_val   = os.listdir(VAL_DIR)

    # Only do splitting if train/val are both empty
    if not existing_train and not existing_val:
        # Gather all image files directly under Images/
        all_images = []
        for fname in os.listdir(IMAGES_DIR):
            fpath = os.path.join(IMAGES_DIR, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTS:
                    all_images.append(fname)

        if not all_images:
            print("❗ No image files found under Images/. Skipping split.")
        else:
            # Pair each image with a label if it exists
            pairs = []
            for img_fname in all_images:
                basename, _ = os.path.splitext(img_fname)
                label_fname = None
                for lext in LABEL_EXTS:
                    candidate = f"{basename}{lext}"
                    if os.path.isfile(os.path.join(IMAGES_DIR, candidate)):
                        label_fname = candidate
                        break
                pairs.append((img_fname, label_fname))

            print(f"Found {len(pairs)} images total (with {sum(1 for _,lbl in pairs if lbl)} labels).")

            # Shuffle & split
            random.shuffle(pairs)
            n_train = int(len(pairs) * TRAIN_RATIO)
            train_pairs = pairs[:n_train]
            val_pairs   = pairs[n_train:]

            print(f"→ {len(train_pairs)} pairs → TRAIN/")
            print(f"→ {len(val_pairs)} pairs → VAL/")

            def move_pair(dst_dir, img_fname, lbl_fname):
                src_img = os.path.join(IMAGES_DIR, img_fname)
                dst_img = os.path.join(dst_dir, img_fname)
                shutil.move(src_img, dst_img)
                if lbl_fname:
                    src_lbl = os.path.join(IMAGES_DIR, lbl_fname)
                    dst_lbl = os.path.join(dst_dir, lbl_fname)
                    shutil.move(src_lbl, dst_lbl)
                else:
                    print(f"⚠️  No label found for image {img_fname}; moved image alone.")

            for img_fname, lbl_fname in train_pairs:
                move_pair(TRAIN_DIR, img_fname, lbl_fname)
            for img_fname, lbl_fname in val_pairs:
                move_pair(VAL_DIR, img_fname, lbl_fname)

            print("✅ Dataset split complete.")
    else:
        print("ℹ️  Images/train or Images/val already contains files; skipping dataset split.")

    # 3) Write data.yaml
    yaml_content = f"""
    train: {TRAIN_DIR}
    val:   {VAL_DIR}

    nc: 2
    names: ['door', 'window']
    """.strip()

    with open(DATA_YAML_PATH, "w") as f:
        f.write(yaml_content)
    print(f"✅ Wrote data.yaml to {DATA_YAML_PATH}:\n{yaml_content}")

    # 4) Kick off YOLOv8 training (if best.pt doesn’t already exist)
    if os.path.exists(BEST_WEIGHTS):
        print(f"📦 Found existing weights at {BEST_WEIGHTS}, skipping training.")
        return

    print("🛠️  Starting YOLOv8 training in background… (this may take a while)")

    # Load pretrained YOLOv8n and train on our data.yaml
    model = YOLO(YOLO_BASE)
    model.train(
        data=DATA_YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        name="door_window_yolov8",
        device="cpu"
    )
    print("✅ Training complete. best.pt should now exist.")

# Launch the above in a daemon thread so Flask can start immediately
train_thread = threading.Thread(target=prepare_and_train, daemon=True)
train_thread.start()

# ─────────────────────────────────────────────────────────────
# FLASK APP FOR INFERENCE
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
_model = None  # will hold our loaded YOLO model

def try_load_model():
    """
    If best.pt exists, load it into a global variable.
    If it’s already loaded, do nothing.
    """
    global _model
    if _model is None and os.path.exists(BEST_WEIGHTS):
        print(f"📦 Loading trained model from {BEST_WEIGHTS}")
        _model = YOLO(BEST_WEIGHTS)
        print("📦 Model loaded. Classes:", _model.names)

# Every 10 seconds, check if best.pt appeared, and load it
def model_loader_loop():
    while True:
        try_load_model()
        time.sleep(10)

# Start a background thread to watch for the trained weights
loader_thread = threading.Thread(target=model_loader_loop, daemon=True)
loader_thread.start()

@app.route("/detect", methods=["POST"])
def detect():
    # If model isn’t loaded yet, return a "try again later" JSON
    if _model is None:
        return jsonify({"error": "Model not yet trained/loaded. Please retry in a minute."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Run prediction
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
# STARTUP
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure Flask listens on 0.0.0.0 and uses Render’s $PORT
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
