import re
import os
import shutil
import cv2
import random

import os
import re


# Function to analyze directories and collect video files
def analyze_directories(directories):
    videos = {}
    pattern = re.compile(r"processed_(\d{6}).*\.mp4")

    for folder in directories:
        if os.path.exists(folder):
            video_count = sum(
                1
                for file in os.listdir(folder)
                if file.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
                )
            )
            print(f"Folder {folder}: {video_count} videos")

            for filename in os.listdir(folder):
                match = pattern.match(filename)
                if match:
                    video_id = f"processed_{match.group(1)}"
                    videos[video_id] = os.path.join(folder, filename)
        else:
            print(f"Folder {folder} does not exist")

    return videos


# Function to compare videos with metadata and update if necessary
def compare_and_update_metadata(metadata_file, videos):
    def read_metadata(file_path):
        metadata = {}
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 1:
                    video_id = parts[0].split(".")[0]
                    metadata[video_id] = line.strip()
        return metadata

    metadata = read_metadata(metadata_file)

    metadata_set = set(metadata.keys())
    videos_set = set(videos.keys())

    missing_in_directories = metadata_set - videos_set
    extra_in_directories = videos_set - metadata_set

    print(
        f"\nVideos in metadata but missing in directories: {len(missing_in_directories)}"
    )
    for vid in sorted(missing_in_directories)[:10]:
        print(f"  {vid}")
    if len(missing_in_directories) > 10:
        print(f"  ... and {len(missing_in_directories) - 10} more")

    print(f"\nVideos in directories but not in metadata: {len(extra_in_directories)}")
    for vid in sorted(extra_in_directories)[:10]:
        print(f"  {vid}")
    if len(extra_in_directories) > 10:
        print(f"  ... and {len(extra_in_directories) - 10} more")

    # Modify the condition to check for both missing and extra videos
    if len(missing_in_directories) == 0 and len(extra_in_directories) == 0:
        print(
            "\nAll videos are correctly matched between directories and metadata. No update necessary."
        )
    else:
        print("\nMismatches found between videos and metadata.")
        # Only proceed to create an updated metadata file if there are missing videos in directories
        if len(missing_in_directories) > 0 and len(extra_in_directories) == 0:
            # Fix the updated metadata file name
            metadata_basename = os.path.basename(metadata_file)
            metadata_name, metadata_ext = os.path.splitext(metadata_basename)
            new_metadata_name = metadata_name + "_updated" + metadata_ext
            new_metadata_file = os.path.join(
                os.path.dirname(metadata_file), new_metadata_name
            )

            with open(new_metadata_file, "w") as f:
                for video_id in sorted(videos_set):
                    if video_id in metadata:
                        f.write(metadata[video_id] + "\n")
            print(f"\nNew metadata file created: {new_metadata_file}")
            print(f"Total videos in new metadata: {len(videos_set)}")
        else:
            print("New metadata file not created due to mismatches.")


# Main execution
directories = [
    "/home/ubuntu/Desktop/eugene/ComfyUI/output/processed_vidoes_1911",
]

metadata_file = (
    "/home/ubuntu/Desktop/eugene/ComfyUI/input/_temp/processed_video_1911.txt"
)

# videos = analyze_directories(directories)
# compare_and_update_metadata(metadata_file, videos)


"""reduce video size if exceeding 4mb"""


# Function to count video files and categorize by size
def analyze_video_files(directory):
    video_extensions = (".mp4", ".mkv", ".mov", ".avi", ".wmv")
    larger_files = []
    larger_count = 0
    smaller_count = 0

    # Iterate through files in the directory
    for file in os.listdir(directory):
        if file.lower().endswith(video_extensions):
            file_path = os.path.join(directory, file)
            file_size = os.path.getsize(file_path)  # Get file size in bytes

            if file_size > 3 * 1000 * 1000:  # Check if larger than 3 MB
                larger_files.append(file)
                larger_count += 1
            else:
                smaller_count += 1

    # Write larger file names to a text file
    with open("_larger_video_files.txt", "w") as f:
        for larger_file in sorted(larger_files):
            f.write(larger_file + "\n")

    return len(larger_files) + smaller_count, larger_count, smaller_count


# Step 0: Analyze video from the text file to find large video (>= 3MB)
try:
    total_videos, larger_count, smaller_count = analyze_video_files(directories[0])
    print(f"Total video files: {total_videos}")
    print(f"Files larger than 3 MB: {larger_count}")
    print(f"Files smaller or equal to 3 MB: {smaller_count}")
except FileNotFoundError:
    print("Directory not found.")


# copy video appears in source text to new destination directory
def copy_files(text_file, source, destination):
    with open(text_file, "r") as file:
        audio_files = file.readlines()

    processed_files = {}
    for audio_file in audio_files:
        audio_file = audio_file.strip()  # Remove any leading/trailing whitespace
        source_file_path = os.path.join(source, audio_file)

        # Check if the source file exists
        if os.path.exists(source_file_path):
            # Copy the file to the new directory
            shutil.copy(source_file_path, destination)
            processed_files[audio_file] = {"status": "copied"}
        else:
            processed_files[audio_file] = {"status": "does not exist"}


# Define the directories
destination_dir = "/home/ubuntu/Desktop/eugene/ComfyUI/output/large_videos"
os.makedirs(destination_dir, exist_ok=True)

# Step 1: Copy video files from the text file
copy_status = copy_files("_larger_video_files.txt", directories[0], destination_dir)

import os
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# Step 2: Reduce video bitrate to a baseline of 5000k for video larger than 3MB

# Check if the directory exists
if os.path.exists(destination_dir):
    # Collect video files
    video_files = [
        file
        for file in os.listdir(destination_dir)
        if file.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    # Function to process a single video file
    def process_video(file):
        video_path = os.path.join(destination_dir, file)
        output_path = os.path.join(destination_dir, f"reduced_{file}")

        # Load the video file
        with VideoFileClip(video_path) as video:
            # Set the target bitrate
            video.write_videofile(output_path, bitrate="5000k")

    # Function to replace original files with reduced bitrate versions
    def replace_original_files():
        for file in video_files:
            original_path = os.path.join(destination_dir, file)
            reduced_path = os.path.join(destination_dir, f"reduced_{file}")

            if os.path.exists(reduced_path):
                os.remove(original_path)
                os.rename(reduced_path, original_path)
            else:
                print(f"Skipping {file}: Reduced file not found.")

    # Function to process videos concurrently
    def process_videos_concurrently():
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(
                tqdm(
                    executor.map(process_video, video_files),
                    total=len(video_files),
                    desc="Processing videos",
                    unit="file",
                )
            )

        replace_original_files()

    process_videos_concurrently()
else:
    print("Directory does not exist.")


"""Select 500 videos for simple demos """


def categorize_descriptions(input_file_path, output_file_path):
    categories = {"0-100": [], "100-200": [], "200-300": [], "others": []}

    with open(input_file_path, "r") as file:
        for line in file:
            name, description = line.strip().split("|", 1)
            length = len(description)

            if length <= 100:
                categories["0-100"].append(line.strip())
            elif length <= 200:
                categories["100-200"].append(line.strip())
            elif length <= 300:
                categories["200-300"].append(line.strip())
            else:
                categories["others"].append(line.strip())

    with open(output_file_path, "w") as out_file:
        for category, lines in categories.items():
            out_file.write(f"Category: {category}\n")
            out_file.write(f"Number of items: {len(lines)}\n\n")
            for line in lines:
                out_file.write(f"{line}\n")
            out_file.write("\n" + "=" * 50 + "\n\n")


def randomly_select_and_copy_videos(input_file_path, output_dir, num_videos=500):
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Read all lines from the input file
        with open(input_file_path, "r") as file:
            all_lines = file.readlines()

        # Randomly select 500 lines
        selected_lines = random.sample(all_lines, min(num_videos, len(all_lines)))

        # Write selected lines to a new file in the output directory
        output_file_path = os.path.join(output_dir, "selected_metadata_500.txt")
        with open(output_file_path, "w") as out_file:
            out_file.writelines(selected_lines)

        print(
            f"Successfully selected {len(selected_lines)} videos and saved metadata to {output_file_path}"
        )

    except FileNotFoundError:
        print(f"Error: The input file {input_file_path} was not found.")
    except PermissionError:
        print(
            f"Error: Permission denied when trying to read {input_file_path} or write to {output_dir}."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


# Example usage
input_file_path = (
    "/home/ubuntu/Desktop/eugene/ComfyUI/input/_complete/data_complete_videos_16730.txt"
)
output_file_path = "/home/ubuntu/Desktop/eugene/ComfyUI/input/_complete/data_complete_videos_16730_categorized.txt"

output_dir = "/home/ubuntu/Desktop/eugene/ComfyUI/input/_complete/"

# categorize_descriptions(input_file_path, output_file_path)
# randomly_select_and_copy_videos(input_file_path, output_dir)

# print(f"Categorized metadata has been written to {output_file_path}")
