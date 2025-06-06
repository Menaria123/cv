import os

def print_directory_tree(root_dir, indent=''):
    try:
        for item in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, item)
            if os.path.isdir(path):
                print(f"{indent}📁 {item}/")
                print_directory_tree(path, indent + '    ')
            else:
                print(f"{indent}📄 {item}")
    except FileNotFoundError:
        print(f"{indent}[Error] Directory not found: {root_dir}")

# Change this to your base directory
base_directory = '.'  # current directory, or e.g., 'Images'
print(f"Directory structure of: {os.path.abspath(base_directory)}\n")
print_directory_tree(base_directory)
