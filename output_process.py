import os
import zipfile
from tqdm import tqdm

# Define the directory and zip file name
directory = "/home/ubuntu/Desktop/eugene/ComfyUI/output/"
zip_file_name = "mixkit_videos_2639.zip"

# Define the range of files
start_number = 1
end_number = 2639

# Create a ZipFile object
with zipfile.ZipFile(zip_file_name, "w") as zipf:
    # Use tqdm to create a progress bar
    for number in tqdm(
        range(start_number, end_number + 1), desc="Zipping files", unit="file"
    ):
        # Construct the file name
        file_name = f"mixkit_v2_{number:05d}.mp4"
        # Get the full path of the file
        file_path = os.path.join(directory, file_name)
        # Check if the file exists before adding
        if os.path.isfile(file_path):
            # Add the file to the zip
            zipf.write(file_path, os.path.basename(file_path))

print(
    f"Video files from mixkit_v2_00001.mp4 to mixkit_v2_02639.mp4 have been zipped into {zip_file_name}."
)
