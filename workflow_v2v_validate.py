import os
import sys
import re


def extract_id(filename):
    match = re.search(r"(\d+)", filename)
    return match.group(1) if match else None


def categorize_files(directory):
    small_files = []  # < 1MB
    medium_files = []  # 1-27 MB
    large_files = []  # > 27 MB

    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
                file_id = extract_id(filename)

                if file_id:
                    if file_size < 1:
                        small_files.append(file_id)
                    elif 1 <= file_size <= 27:
                        medium_files.append(file_id)
                    else:
                        large_files.append(file_id)

        return small_files, medium_files, large_files

    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to access directory '{directory}'.")
        return None


def process_metadata(metadata_file, medium_ids, small_large_output, final_output):
    small_large_count = 0
    medium_count = 0

    with open(metadata_file, "r") as f, open(small_large_output, "w") as sl_out, open(
        final_output, "w"
    ) as final_out:
        for line in f:
            file_id = line.split("|")[0].split("_")[1].split(".")[0]
            if file_id in medium_ids:
                final_out.write(line)
                medium_count += 1
            else:
                sl_out.write(line)
                small_large_count += 1

    return small_large_count, medium_count


if __name__ == "__main__":
    batch = 21
    input_directory = f"input/bs1000_b{batch}/latent_images"
    metadata_file = f"input/bs1000_b{batch}/metadata_bs1000_b{batch}.txt"
    small_large_output = f"input/bs1000_b{batch}/metadata_bs1000_b{batch}_mismatch.txt"
    final_output = f"input/bs1000_b{batch}/metadata_final_b{batch}.txt"
    result = categorize_files(input_directory)

    if result:
        small, medium, large = result
        total = len(small) + len(medium) + len(large)
        print(f"File size categories in '{input_directory}':")
        print(f"Small files (< 1 MB): {len(small)}")
        print(f"Medium files (1-27 MB): {len(medium)}")
        print(f"Large files (> 27 MB): {len(large)}")
        print(f"Total files: {total}")

        medium_ids = set(medium)
        small_large_count, medium_count = process_metadata(
            metadata_file, medium_ids, small_large_output, final_output
        )

        print(f"\nWrote {small_large_count} lines to {small_large_output}")
        print(f"Wrote {medium_count} lines to {final_output}")
        print(f"These lines corresponded to {len(medium_ids)} medium files")
