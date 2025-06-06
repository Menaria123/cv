import os
import random
import shutil

# ──── CONFIG ────
BASE_DIR   = os.getcwd()                   # /opt/render/project/src
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
TRAIN_DIR  = os.path.join(IMAGES_DIR, "train")
VAL_DIR    = os.path.join(IMAGES_DIR, "val")

# Supported image and label extensions:
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXTS = {".txt"}  # assuming YOLO-format .txt labels

# Percentage split (0.8 = 80% train, 20% val)
TRAIN_RATIO = 0.8

# ──── ENSURE TARGET FOLDERS EXIST ────
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR,   exist_ok=True)

# ──── STEP 1: Gather all image files under Images/ (excluding train/ and val/ subfolders) ────
all_images = []
for fname in os.listdir(IMAGES_DIR):
    fpath = os.path.join(IMAGES_DIR, fname)
    if os.path.isfile(fpath):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            all_images.append(fname)

if not all_images:
    print("❗ No image files found directly under Images/. Make sure your .png/.jpg files are there.")
    exit(1)

# ──── STEP 2: For each image, check if a same-basename .txt exists ────
# If not found, we'll still move the image but warn (YOLOv8 can train on images without labels,
# it will just consider them “no objects,” which may or may not be what you want.)
pairs = []
for img_fname in all_images:
    basename, _ = os.path.splitext(img_fname)
    # Look for ANY matching label extension in Images/
    label_fname = None
    for lbl_ext in LABEL_EXTS:
        candidate = f"{basename}{lbl_ext}"
        if os.path.isfile(os.path.join(IMAGES_DIR, candidate)):
            label_fname = candidate
            break
    pairs.append((img_fname, label_fname))

print(f"Found {len(pairs)} images total (with {sum(1 for _,lbl in pairs if lbl):d} labels).")

# ──── STEP 3: Shuffle & split into train/val ────
random.shuffle(pairs)
n_train = int(len(pairs) * TRAIN_RATIO)
train_pairs = pairs[:n_train]
val_pairs   = pairs[n_train:]

print(f"→ {len(train_pairs)} pairs → TRAIN/")
print(f"→ {len(val_pairs)} pairs → VAL/")

# ──── STEP 4: Move files into Images/train/ and Images/val/ ────
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

print("✅ Done. Check that Images/train/ and Images/val/ are now populated.")

# ──── BONUS: Print directory trees for verification ────
def print_tree(root, prefix=""):
    items = sorted(os.listdir(root))
    for i, name in enumerate(items):
        path = os.path.join(root, name)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + name + ("/" if os.path.isdir(path) else ""))
        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

print("\nCurrent folder structure under Images/:")
print(f"Images/")
print_tree(IMAGES_DIR, prefix="    ")
