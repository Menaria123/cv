import os

# Get current working directory
cwd = os.getcwd()

# Join to get the full path to the Images folder
images_path = os.path.join(cwd, "Images")

# Print the full path
print("Absolute path to 'Images' folder:", images_path)
