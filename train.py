import os

# Define supported file types
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
label_extensions = ['.txt', '.xml']

def find_images_and_labels(base_dir):
    image_label_pairs = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_path = os.path.join(root, file)

                # Try to find a matching label file (same name, different extension)
                basename = os.path.splitext(file)[0]
                possible_labels = [
                    os.path.join(root, basename + ext) for ext in label_extensions
                ]

                # Check in a "labels" subdirectory if not found in current folder
                label_found = None
                for label_path in possible_labels:
                    if os.path.exists(label_path):
                        label_found = label_path
                        break

                # Look in a sibling 'labels' folder (e.g., Images/train -> labels/train)
                if not label_found:
                    labels_dir = root.replace('images', 'labels')
                    for ext in label_extensions:
                        label_candidate = os.path.join(labels_dir, basename + ext)
                        if os.path.exists(label_candidate):
                            label_found = label_candidate
                            break

                image_label_pairs.append((image_path, label_found))

    return image_label_pairs


# Example usage
dataset_dir = 'Images'  # or wherever your dataset is
pairs = find_images_and_labels(dataset_dir)

for img, lbl in pairs:
    print(f"Image: {img}")
    if lbl:
        print(f" Label: {lbl}")
    else:
        print(" Label: ❌ Not found")
    print('-' * 40)
