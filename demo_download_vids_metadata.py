#!/usr/bin/env python3
"""
Fetch metadata for a single YouTube video ID: HsXS1Qt11cU. This is for the demo for the presentation at AISDC
Assumes `challenge_data.csv` already exists and contains `youtube_url` and `duration` columns.
Outputs:
 - video_full_metadata.csv
 - video_errors.csv
 - `videos_full/` directory (metadata-only; no download performed by default)
"""

import os
import pandas as pd
from pytubefix import YouTube  # fixed pytube client
from pytubefix.cli import on_progress

VIDEO_ID = "HsXS1Qt11cU"
YOUTUBE_URL = f"https://www.youtube.com/shorts/HsXS1Qt11cU"
VIDEO_FOLDER = "videos_full"

# Ensure output folder exists
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Data collectors
video_data = []
failed_urls = []

# Load challenge data
print("Loading challenge_data.csv...")
try:
    df = pd.read_csv("challenge_data.csv", dtype=str, keep_default_na=False)
except Exception as e:
    print(f"Error loading challenge_data.csv: {e}")
    exit(1)

# Filter for the specific video
mask = df["youtube_url"].str.contains(VIDEO_ID, na=False)
if not mask.any():
    print(f"No entries found in challenge_data.csv for video ID {VIDEO_ID}")
    exit(1)

row = df[mask].iloc[0]
duration = row.get("duration", "")

print(f"Processing metadata for video ID: {VIDEO_ID}")
try:
    # Initialize YouTube client
    yt = YouTube(YOUTUBE_URL, on_progress_callback=on_progress)

    # Fetch basic metadata
    title = yt.title
    description = yt.description

    # Retrieve captions if available
    caption_text = ""
    caption_lang = ""
    try:
        if yt.captions:
            if "en" in yt.captions:
                caption_text = yt.captions["en"].generate_srt_captions()
                caption_lang = "en"
            elif "a.en" in yt.captions:
                caption_text = yt.captions["a.en"].generate_srt_captions()
                caption_lang = "a.en"
            else:
                first_code = list(yt.captions.keys())[0]
                caption_text = yt.captions[first_code].generate_srt_captions()
                caption_lang = first_code
    except Exception as e:
        print(f"Warning: error fetching captions: {e}")

    # Remaining metadata
    publish_date = yt.publish_date
    rating = yt.rating
    channel_id = yt.channel_id
    channel_url = yt.channel_url
    thumbnail_url = yt.thumbnail_url
    channel_name = yt.author
    views = yt.views
    keywords = yt.keywords  # list
    actual_video_id = yt.video_id  # confirmed by client

    # Compile metadata record
    record = {
        "youtube_url": YOUTUBE_URL,
        "title": title,
        "description": description,
        "caption": caption_text,
        "caption_lang": caption_lang,
        "publish_date": publish_date,
        "rating": rating,
        "channel_id": channel_id,
        "channel_url": channel_url,
        "thumbnail_url": thumbnail_url,
        "channel_name": channel_name,
        "views": views,
        "keywords": keywords,
        "duration": duration,
        "video_path": os.path.join(VIDEO_FOLDER, f"{VIDEO_ID}.mp4"),
        "video_id": actual_video_id,
    }
    video_data.append(record)
    print(f"Metadata fetched successfully for {VIDEO_ID}")

except Exception as e:
    print(f"Error processing video {VIDEO_ID}: {e}")
    failed_urls.append({"youtube_url": YOUTUBE_URL, "error": str(e)})

# Save outputs
print("Saving metadata to video_full_metadata.csv...")
pd.DataFrame(video_data).to_csv(
    "video_full_metadata.csv", index=False, encoding="utf-8-sig"
)
# print("Saving errors to video_errors.csv...")
pd.DataFrame(failed_urls).to_csv("video_errors.csv", index=False, encoding="utf-8-sig")

print("Done.")
