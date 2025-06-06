import os
import yaml

# Set the root dataset folder
dataset_root = r"src" 

# Set class names here:
class_names = ['door', 'window']  # Edit this list based on your actual classes

# Paths to training and validation images
train_path = os.path.join(dataset_root, "Images/train")
val_path = os.path.join(dataset_root, "Images/val")

# Relative paths for YAML
yaml_dict = {
    'train': os.path.relpath(train_path, dataset_root).replace("\\", "/"),
    'val': os.path.relpath(val_path, dataset_root).replace("\\", "/"),
    'nc': len(class_names),
    'names': class_names
}

# Save YAML
yaml_file = os.path.join(dataset_root, "data.yaml")
with open(yaml_file, 'w') as f:
    yaml.dump(yaml_dict, f, default_flow_style=False)

print(f"`data.yaml` created at: {yaml_file}")
