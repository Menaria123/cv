import os

print("PWD:", os.getcwd())
print("Files in current dir:", os.listdir('.'))
print("Files in images/val:", os.listdir('./images/val') if os.path.exists('./images/val') else "No val folder")
print("Files in images/train:", os.listdir('./images/train') if os.path.exists('./images/train') else "No train folder")
