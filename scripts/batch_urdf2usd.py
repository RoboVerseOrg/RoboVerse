#!/usr/bin/env python3

import os
import subprocess
import sys


def find_urdf_files(directory):
    """Recursively find all .urdf files in directory and its subdirectories."""
    urdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".urdf"):
                urdf_files.append(os.path.join(root, file))
    return urdf_files


def convert_urdf_to_usd(urdf_path, output_dir):
    """Convert a single URDF file to USD."""
    rel_path = os.path.relpath(urdf_path, args.input_dir)
    output_path = os.path.join(output_dir, rel_path)
    output_path = os.path.splitext(output_path)[0] + ".usd"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nProcessing: {urdf_path}")
    print(f"Output to: {output_path}")

    try:
        subprocess.run(
            [sys.executable, "scripts/urdf2usd.py", urdf_path, output_path, "--merge-joints", "--fix-base"], check=True
        )
        print(f"Successfully converted {urdf_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {urdf_path}: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch convert URDF files to USD format")
    parser.add_argument("input_dir", type=str, help="Input directory containing URDF files")
    parser.add_argument("output_dir", type=str, help="Output directory for USD files")

    global args
    args = parser.parse_args()

    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    urdf_files = find_urdf_files(args.input_dir)

    if not urdf_files:
        print(f"No URDF files found in {args.input_dir}")
        return

    print(f"Found {len(urdf_files)} URDF files to process")

    success_count = 0
    for urdf_file in urdf_files:
        if convert_urdf_to_usd(urdf_file, args.output_dir):
            success_count += 1

    print("\nConversion Summary:")
    print(f"Total files: {len(urdf_files)}")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {len(urdf_files) - success_count}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_urdf2usd.py <input_directory> <output_directory>")
        sys.exit(1)

    main()
