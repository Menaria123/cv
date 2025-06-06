yaml_content = """
train: Images/train
val: Images/val
nc: 2
names: ['door', 'window']
"""

with open('data.yaml', 'w') as f:
    f.write(yaml_content.strip())

print("data.yaml file has been created/updated successfully.")
