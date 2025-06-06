import os

# Get absolute path to Images folder
cwd = os.getcwd()
images_path = os.path.join(cwd, "Images")

# Write full paths to data.yaml
yaml_content = f"""
train: {images_path}/train
val: {images_path}/val
nc: 2
names: ['door', 'window']
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content.strip())

print("data.yaml file created with full image paths.")
