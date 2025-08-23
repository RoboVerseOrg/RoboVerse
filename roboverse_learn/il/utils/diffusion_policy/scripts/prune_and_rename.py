import os
import sys
import shutil

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_directory>")
        sys.exit(1)

    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory.")
        sys.exit(1)

    # List subdirectories matching 'demo_XXXX'
    subdirs = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('demo_')]
    # Sort by numeric suffix
    subdirs.sort(key=lambda x: int(x.split('_')[1]))

    valid_dirs = []
    # Identify and remove empty ones
    for d in subdirs:
        path = os.path.join(root_dir, d)
        metadata_path = os.path.join(path, 'metadata.json')
        if not os.path.isfile(metadata_path):
            print(f"Removing empty folder: {d}")
            shutil.rmtree(path)
        else:
            valid_dirs.append(d)

    # Renumber remaining directories
    for new_idx, old_name in enumerate(valid_dirs):
        new_name = f"demo_{new_idx:04d}"
        if old_name != new_name:
            old_path = os.path.join(root_dir, old_name)
            new_path = os.path.join(root_dir, new_name)
            print(f"Renaming {old_name} -> {new_name}")
            os.rename(old_path, new_path)

if __name__ == '__main__':
    main()
