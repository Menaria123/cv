import os

base = os.getcwd()  # /opt/render/project/src
train_dir = os.path.join(base, "Images", "train")
val_dir   = os.path.join(base, "Images", "val")

print("Checking dataset folders:")
print(f"  Images/train exists? {os.path.isdir(train_dir)}")
print(f"    Contents: {os.listdir(train_dir) if os.path.isdir(train_dir) else 'N/A'}")
print(f"  Images/val   exists? {os.path.isdir(val_dir)}")
print(f"    Contents: {os.listdir(val_dir) if os.path.isdir(val_dir) else 'N/A'}")
