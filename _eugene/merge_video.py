from moviepy.editor import VideoFileClip, concatenate_videoclips

# Load your video clips
clip1 = VideoFileClip(
    "/home/ubuntu/Desktop/Eugene/ComfyUI/output/AnimateDiff_00001.mp4"
)
clip2 = VideoFileClip(
    "/home/ubuntu/Desktop/Eugene/ComfyUI/output/AnimateDiff_00002.mp4"
)

final_clip = concatenate_videoclips([clip1, clip2])

# Write the result to a file
final_clip.write_videofile(
    "/home/ubuntu/Desktop/Eugene/ComfyUI/output/AnimateDiff_00003.mp4"
)
