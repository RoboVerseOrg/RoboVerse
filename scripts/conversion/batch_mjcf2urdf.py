import os
import sys
from datetime import datetime
from pathlib import Path

from mjcf2urdf import mjcf_to_urdf


class ConversionResult:
    def __init__(self, input_file, success, error_msg=None):
        self.input_file = input_file
        self.success = success
        self.error_msg = error_msg
        self.output_file = None


def process_folder(folder_path):
    """
    Recursively process all MJCF/XML files in the given folder and its subfolders.

    Args:
        folder_path (str): Path to the folder containing MJCF/XML files

    Returns:
        list[ConversionResult]: List of conversion results
    """
    results = []
    folder_path = Path(folder_path)

    xml_files = list(folder_path.rglob("*.xml")) + list(folder_path.rglob("*.mjcf"))
    xml_files.sort()

    for input_file in xml_files:
        result = ConversionResult(str(input_file), False)

        try:
            output_file = input_file.with_suffix(".urdf")
            result.output_file = str(output_file)

            output_file.parent.mkdir(parents=True, exist_ok=True)

            mjcf_to_urdf(str(input_file), str(output_file))

            result.success = True
            print(f"Successfully converted: {input_file}")

        except Exception as e:
            result.error_msg = str(e)
            print(f"Failed to convert {input_file}: {e!s}")

        results.append(result)

    return results


def generate_log(results, folder_path):
    """
    Generate a log file with conversion results.
    """
    log_file = os.path.join(folder_path, f"conversion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_file, "w") as f:
        f.write("MJCF to URDF Conversion Log\n")
        f.write("=========================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Root Folder: {folder_path}\n")
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"Successful conversions: {sum(1 for r in results if r.success)}\n")
        f.write(f"Failed conversions: {sum(1 for r in results if not r.success)}\n\n")

        results_by_folder = {}
        for result in results:
            folder = os.path.dirname(result.input_file)
            if folder not in results_by_folder:
                results_by_folder[folder] = []
            results_by_folder[folder].append(result)

        f.write("Detailed Results by Folder:\n")
        f.write("-------------------------\n")
        for folder, folder_results in sorted(results_by_folder.items()):
            f.write(f"\nFolder: {folder}\n")
            f.write("Files:\n")
            for result in folder_results:
                f.write(f"\nInput: {os.path.basename(result.input_file)}\n")
                if result.success:
                    f.write("Status: SUCCESS\n")
                    f.write(f"Output: {os.path.basename(result.output_file)}\n")
                else:
                    f.write("Status: FAILED\n")
                    f.write(f"Error: {result.error_msg}\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_mjcf2urdf.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)

    print(f"Processing files in {folder_path} and its subfolders...")

    results = process_folder(folder_path)
    generate_log(results, folder_path)

    success_count = sum(1 for r in results if r.success)
    total_count = len(results)
    print("\nConversion complete!")
    print(f"Successfully converted: {success_count}/{total_count} files")
    print(f"Check the conversion_log_*.txt file in {folder_path} for details")


if __name__ == "__main__":
    main()
