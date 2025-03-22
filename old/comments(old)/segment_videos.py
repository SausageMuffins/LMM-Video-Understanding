import os
import shutil  # Library to copy files
import pandas as pd  # Library to handle CSV files
from moviepy.video.io.VideoFileClip import (
    VideoFileClip,
)  # Library to process video files


def segment_video(video_path, output_dir, video_id, k=3):
    """
    Segments the input video into k equal parts and saves each segment as an MP4 file.

    Parameters:
        video_path (str): Path to the source video file.
        output_dir (str): Directory where the segmented videos will be saved.
        video_id (str): The ID of the video used in naming the segments.
        k (int): Number of segments to split the video into.
    """
    clip = VideoFileClip(video_path)  # Open the video file
    duration = clip.duration  # Total duration of the video in seconds
    segment_duration = duration / k  # Duration of each segment

    for i in range(k):
        start_time = i * segment_duration
        # Ensure the last segment ends exactly at the end of the video
        end_time = (i + 1) * segment_duration if i < k - 1 else duration
        try:
            subclip = clip.subclip(
                start_time, end_time
            )  # Create a video segment using subclip
        except AttributeError:
            # Fallback: use 'subclipped' if 'subclip' is not available (version-specific issue)
            subclip = clip.subclipped(start_time, end_time)
        output_filename = os.path.join(output_dir, f"{video_id}_{i+1}.mp4")
        subclip.write_videofile(output_filename, codec="libx264")  # Save the segment
    clip.close()  # Close the video file


def main():
    # Read the CSV file containing video metadata
    csv_file = "video_metadata.csv"
    metadata = pd.read_csv(csv_file)  # Reads the CSV file into a DataFrame
    # Create a dictionary to map video titles to their corresponding YouTube URLs
    title_to_url = dict(zip(metadata["title"], metadata["youtube_url"]))

    # Directory containing the .mp4 video files
    videos_dir = "videos"
    processed_video_ids = []  # List to store processed video IDs
    failed_videos = []  # List to store any errors encountered

    # Iterate over each file in the videos directory
    for file in os.listdir(videos_dir):
        if file.endswith(".mp4"):
            video_title = os.path.splitext(file)[
                0
            ]  # Assume video title is the filename without extension

            # Check if metadata for the video is available
            if video_title not in title_to_url:
                failed_videos.append(
                    f"Title '{video_title}' not found in CSV metadata."
                )
                continue

            youtube_url = title_to_url[video_title]
            if not youtube_url or not isinstance(youtube_url, str):
                failed_videos.append(
                    f"YouTube URL not found or invalid for title '{video_title}'."
                )
                continue

            # Extract the video ID by splitting the URL and taking the last portion
            video_id = youtube_url.rstrip("/").split("/")[-1]
            if not video_id:
                failed_videos.append(
                    f"Could not extract video ID from URL '{youtube_url}' for title '{video_title}'."
                )
                continue

            try:
                # Create a new directory for segmented videos, named as {video_id}_k
                output_dir = os.path.join(videos_dir, f"{video_id}_k")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)  # Create the directory if it does not exist

                # Construct the full path of the video file
                video_path = os.path.join(videos_dir, file)
                # Save the original video with full length as {video_id}_0.mp4
                original_output = os.path.join(output_dir, f"{video_id}_0.mp4")
                shutil.copy(video_path, original_output)  # Copy the original video file

                # Segment the video into k equal parts
                segment_video(video_path, output_dir, video_id, k=3)
                processed_video_ids.append(video_id)
            except Exception as e:
                failed_videos.append(
                    f"Error processing video '{video_title}': {str(e)}"
                )
                continue

    # Print the total number of unique YouTube URLs processed
    unique_video_ids = set(processed_video_ids)
    print("Total processed YouTube URLs:", len(unique_video_ids))

    # Print any errors encountered during processing
    if failed_videos:
        print("\nThe following videos encountered errors:")
        for error in failed_videos:
            print(error)


if __name__ == "__main__":
    main()
