import csv  # For CSV operations
import os  # For file and directory operations
import yt_dlp  # For downloading YouTube videos using yt-dlp
from moviepy.video.io.VideoFileClip import (
    VideoFileClip,
)  # For video processing with MoviePy


def split_video_moviepy(downloaded_video_path, video_id, video_dir, k=3):
    """
    Split video into k segments using MoviePy.

    Parameters:
        downloaded_video_path (str): Path to the downloaded video.
        video_id (str): Unique identifier for the video.
        video_dir (str): Directory where the segments will be saved.
        k (int): Number of segments to split the video into.
    """
    try:
        clip = VideoFileClip(downloaded_video_path)  # Open the video file
    except Exception as e:
        print(f"Failed to open video {downloaded_video_path}: {e}")
        return

    total_duration = clip.duration  # Total duration in seconds
    segment_duration = total_duration / k  # Duration of each segment

    for i in range(k):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration if i < k - 1 else total_duration
        try:
            subclip = clip.subclip(start_time, end_time)  # Create a video segment
        except AttributeError:
            subclip = clip.subclipped(start_time, end_time)
        segment_path = os.path.join(video_dir, f"{video_id}_{i+1}.mp4")
        subclip.write_videofile(
            segment_path, codec="libx264", audio_codec="aac"
        )  # Save segment
        print(f"Segment {i+1} saved to {segment_path}")
    clip.close()


def download_and_split_video(video_id, youtube_url, k=3):
    """
    Download a YouTube video using yt-dlp and split it into k segments using MoviePy.

    Parameters:
        video_id (str): Unique identifier for the video.
        youtube_url (str): The YouTube URL.
        k (int): Number of segments to split the video into.
    """
    # Create a directory for the video
    video_dir = os.path.join("videos", video_id)
    os.makedirs(video_dir, exist_ok=True)

    # Set the output path and file name for the downloaded (full-length) video
    downloaded_video_path = os.path.join(video_dir, f"{video_id}_0.mp4")

    # yt-dlp options for downloading the video.
    # Option 1 (requires ffmpeg): "bestvideo+bestaudio/best"
    # Option 2 (avoid merging, no ffmpeg needed): "best[ext=mp4]"
    ydl_opts = {
        "outtmpl": downloaded_video_path,  # Output template
        "format": "best[ext=mp4]",  # Download best available mp4 format to avoid merging
        "merge_output_format": "mp4",  # Ensure output is mp4
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        print(f"Failed to download from URL {youtube_url}: {e}")
        record_failure(youtube_url)
        return

    # Split the downloaded video into k segments using MoviePy
    split_video_moviepy(downloaded_video_path, video_id, video_dir, k)


def record_failure(failing_entry):
    """
    Record a failing YouTube URL or message into 'videos/failed_links.csv'.

    Parameters:
        failing_entry (str): The YouTube URL or error message that failed.
    """
    failed_csv = os.path.join("videos", "failed_links.csv")
    with open(failed_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([failing_entry])


def main():
    # Ensure 'videos' directory exists
    os.makedirs("videos", exist_ok=True)

    failed_links = []  # To store failure messages for summary printing

    # Open the CSV file and process each row
    with open("challenge_data.csv", mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_id = row.get("video_id", "").strip()
            youtube_url = row.get("youtube_url", "").strip()
            if not video_id or not youtube_url:
                error_message = f"Missing video_id or youtube_url for row: {row}"
                print(error_message)
                record_failure(youtube_url if youtube_url else "Missing URL")
                failed_links.append(error_message)
                continue
            download_and_split_video(video_id, youtube_url, k=3)

    # Print summary of failed entries, if any
    if failed_links:
        print("\nThe following entries encountered errors:")
        for error in failed_links:
            print(error)


if __name__ == "__main__":
    main()
