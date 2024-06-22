import os
import zipfile
from tqdm import tqdm
import shutil
import random

# Define the directory and zip file name
BASE_DIRECTORY = "/home/ubuntu/Desktop/eugene/ComfyUI/output/"
ZIP_FILE_NAME = "mixkit_videos_2639.zip"
NEW_DIRECTORY = "/home/ubuntu/Desktop/eugene/ComfyUI/mixkit_videos_2639/"

# Ensure the new directory exists
os.makedirs(NEW_DIRECTORY, exist_ok=True)

# Define the range of files
START_NUMBER = 1
END_NUMBER = 2639
TOTAL_VIDEOS = 100

INPUT_FILE_PATH = "/home/ubuntu/Desktop/eugene/ComfyUI/input/__mixkit_v2__/data_c.txt"
OUTPUT_FILE_PATH = "/home/ubuntu/Desktop/eugene/ComfyUI/data_mixkit_videos_2639.txt"
LINES_TO_CUT = 2639


def zip_files(base_directory, zip_file, start_num, end_num):
    with zipfile.ZipFile(zip_file, "w") as zipf:
        for number in tqdm(
            range(start_num, end_num + 1), desc="Zipping files", unit="file"
        ):
            file_name = f"mixkit_v2_{number:05d}.mp4"
            file_path = os.path.join(base_directory, file_name)
            if os.path.isfile(file_path):
                zipf.write(file_path, os.path.basename(file_path))
    print(
        f"Video files from mixkit_v2_00001.mp4 to mixkit_v2_02639.mp4 have been zipped into {zip_file}."
    )


def select_random_videos(
    base_directory, new_directory, start_num, end_num, total_videos
):
    selected_files = set()
    with tqdm(total=total_videos, desc="Copying files", unit="file") as pbar:
        while len(selected_files) < total_videos:
            number = random.randint(start_num, end_num)
            file_name = f"mixkit_v2_{number:05d}.mp4"
            file_path = os.path.join(base_directory, file_name)
            if os.path.isfile(file_path) and file_name not in selected_files:
                selected_files.add(file_name)
                shutil.copy(file_path, os.path.join(new_directory, file_name))
                pbar.update(1)
    print(
        f"Randomly selected {total_videos} video files have been copied to {new_directory}."
    )


def count_videos_in_directory(directory_path):
    video_count = sum(
        1 for file_name in os.listdir(directory_path) if file_name.endswith(".mp4")
    )
    return video_count


def find_missing_videos(base_directory, start_num, end_num):
    expected_files = {
        f"mixkit_v2_{number:05d}.mp4" for number in range(start_num, end_num + 1)
    }
    existing_files = {
        file_name
        for file_name in os.listdir(base_directory)
        if file_name.startswith("mixkit_v2_") and file_name.endswith(".mp4")
    }
    missing_files = expected_files - existing_files
    return missing_files


def cut_and_format_file_lines(input_path, output_path, num_lines):
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for i in range(1, num_lines + 1):
            line = infile.readline()
            if not line:
                break
            parts = line.strip().split("|")
            if len(parts) == 3:
                description = parts[2]
                formatted_line = f"mixkit_v2_{i:05d}.mp4|{description}"
                outfile.write(formatted_line + "\n")
    print(
        f"The first {num_lines} lines have been cut, formatted, and saved to {output_path}."
    )


def main():
    # Uncomment the desired function calls

    # zip_files(BASE_DIRECTORY, ZIP_FILE_NAME, START_NUMBER, END_NUMBER)

    # select_random_videos(BASE_DIRECTORY, NEW_DIRECTORY, START_NUMBER, END_NUMBER, TOTAL_VIDEOS)

    # video_count = count_videos_in_directory(NEW_DIRECTORY)
    # print(f"Total videos in {NEW_DIRECTORY}: {video_count}")

    # missing_videos = find_missing_videos(BASE_DIRECTORY, START_NUMBER, END_NUMBER)
    # if missing_videos:
    #     print(f"Missing videos: {sorted(missing_videos)}")
    # else:
    #     print("No missing videos found.")

    cut_and_format_file_lines(INPUT_FILE_PATH, OUTPUT_FILE_PATH, LINES_TO_CUT)


if __name__ == "__main__":
    main()
