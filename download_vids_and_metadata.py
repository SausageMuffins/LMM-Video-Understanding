#!/usr/bin/env python3

import os
import pandas as pd
from datasets import load_dataset
# from pytube import YouTube # Original comment kept
from pytubefix import YouTube
from pytubefix.cli import on_progress

# --- Load dataset from Hugging Face ---
print("Loading dataset from Hugging Face (requires login)...")
# Login using e.g. `huggingface-cli login` to access this dataset
try:
    ds = load_dataset("lmms-lab/AISG_Challenge")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you are logged in using 'huggingface-cli login' and have access.")
    exit(1)

# --- Convert dataset to DataFrame and save ---
print("Converting dataset to DataFrame...")
# save ds as a csv
df = pd.DataFrame(ds['test'])
# df # Removed display line typical of notebooks

# save df
print("Saving initial challenge data to challenge_data.csv...")
df.to_csv('challenge_data.csv', index=False, encoding='utf-8-sig')

# --- Process YouTube URLs for Metadata ---
print("Processing YouTube URLs to fetch metadata...")

# Example: assume you already have df loaded with columns: ['youtube_url', 'duration']

# 1. Drop duplicate URLs
df_unique = df.drop_duplicates(subset=['youtube_url']).reset_index(drop=True)

# Lists to collect successful data and errors
video_data = []
failed_urls = []

video_folder = 'videos_full'
print(f"Ensuring video directory exists: {video_folder}")
os.makedirs(video_folder, exist_ok=True)

# 2. Iterate over each unique URL
print(f"Iterating over {len(df_unique)} unique URLs...")
for idx, row in df_unique.iterrows():
    yt_url = row['youtube_url']
    duration = row['duration']  # from your original DataFrame

    # Handle potential missing or non-string URLs gracefully
    if not isinstance(yt_url, str) or not yt_url.strip():
        print(f"Skipping row {idx} due to invalid URL: {yt_url}")
        error_info = {
            'youtube_url': yt_url,
            'error': 'Invalid or missing URL in input data'
        }
        failed_urls.append(error_info)
        continue

    # --- Video ID Extraction Logic ---
    video_id_part = yt_url.strip().split("/")[-1]

    # Refinement for standard watch URLs (e.g., .../watch?v=ID&...)
    if '?v=' in video_id_part:
        # Extract the part after 'v='
        video_id = video_id_part.split('v=')[-1]
        # Remove any subsequent query parameters (like &t=, &list=)
        if '&' in video_id:
            video_id = video_id.split('&')[0]
    else:
        # If '?v=' is not present, assume the last part is the ID (as per original logic)
        video_id = video_id_part

    # Basic check if extraction resulted in an empty string or clearly invalid chars
    if not video_id or not video_id.isalnum() and '_' not in video_id and '-' not in video_id:
        print(f"Warning: Possibly invalid video ID extracted ('{video_id}') from {yt_url}. Using '{video_id_part}' as fallback for path.")
        video_id = video_id_part # Fallback for path, metadata will use yt.video_id later if successful
        if not video_id: # If fallback is also empty
             print(f"ERROR: Could not determine a video ID for URL {yt_url}. Skipping.")
             error_info = {
                'youtube_url': yt_url,
                'error': 'Could not extract video ID'
             }
             failed_urls.append(error_info)
             continue # Skip this URL entirely if no ID could be guessed

    video_path = os.path.join(video_folder, f"{video_id}.mp4")
    # --- End Video ID Extraction Logic ---


    print(f"\nProcessing ({idx+1}/{len(df_unique)}): {yt_url}")
    try:
        yt = YouTube(yt_url, on_progress_callback=on_progress)

        # Collect metadata in order of importance
        print(f"  Fetching metadata for {yt_url}...")
        title         = yt.title
        description   = yt.description
        # 2a. Retrieve captions in a fallback manner
        caption_text = ''
        caption_lang = ''
        try:
            if yt.captions:  # If there are any captions at all
                # Prefer English if available
                if 'en' in yt.captions:
                    caption_text = yt.captions['en'].generate_srt_captions()
                    caption_lang = 'en'
                elif 'a.en' in yt.captions:
                    caption_text = yt.captions['a.en'].generate_srt_captions()
                    caption_lang = 'a.en'
                else:
                    # Fallback: pick the first available language
                    first_lang = list(yt.captions.keys())[0].code
                    caption_text = yt.captions[first_lang].generate_srt_captions()
                    caption_lang = first_lang
        except Exception as e:
            print(f"  Error fetching captions: {e}")
            caption_text = ''
            caption_lang = ''

        print(f"  Fetching remaining metadata...")
        publish_date  = yt.publish_date
        rating        = yt.rating
        channel_id    = yt.channel_id
        channel_url   = yt.channel_url
        thumbnail_url = yt.thumbnail_url
        channel_name  = yt.author
        views         = yt.views
        keywords      = yt.keywords  # list of strings
        actual_video_id = yt.video_id # Use the ID confirmed by pytubefix

        # 3. Download the video (Commented out as per original)
        # print(f"  Downloading video (optional, currently commented out)...")
        # stream = yt.streams.get_highest_resolution()
        # video_path = stream.download(output_path=download_folder)
        # Change video_path to the relative path
        # path = download_folder + '/' + video_id


        # 4. Store metadata
        info = {
            'youtube_url':   yt_url,
            'title':         title,
            'description':   description,
            'caption':       caption_text,
            'caption_lang':  caption_lang,
            'publish_date':  publish_date,
            'rating':        rating,
            'channel_id':    channel_id,
            'channel_url':   channel_url,
            'thumbnail_url': thumbnail_url,
            'channel_name':  channel_name,
            'views':         views,
            'keywords':      keywords,
            'duration':      duration,
            'video_path':    video_path, # Path constructed earlier based on extracted ID
            'video_id':      actual_video_id # ID confirmed by pytubefix
        }

        video_data.append(info)
        print(f"  Processed URL {yt_url} successfully.")

    except Exception as e:
        # If anything fails, capture the URL and reason
        error_info = {
            'youtube_url': yt_url,
            'error': str(e)
        }
        failed_urls.append(error_info)
        print(f"  Failed to process URL {yt_url}: {e}")

        # Store placeholder info when processing fails
        info = {
            'youtube_url':   yt_url,
            'title':         None,
            'description':   None,
            'caption':       None,
            'caption_lang':  None,
            'publish_date':  None,
            'rating':        None,
            'channel_id':    None,
            'channel_url':   None,
            'thumbnail_url': None,
            'channel_name':  None,
            'views':         None,
            'keywords':      None,
            'duration':      duration, # Keep original duration if available
            'video_path':    video_path, # Keep constructed path
            'video_id':      video_id # Keep extracted ID as fallback
        }
        video_data.append(info) # Still add row to metadata, but with Nones and error logged

# --- Create and Save Output DataFrames ---
print("\nCreating output DataFrames...")
# 5. Create DataFrames from successful and failed entries
metadata_df = pd.DataFrame(video_data)
failed_df = pd.DataFrame(failed_urls)

# 6. Save both DataFrames to CSV in the same location
print("Saving metadata to video_full_metadata.csv...")
metadata_df.to_csv('video_full_metadata.csv', index=False, encoding='utf-8-sig')

print("Saving errors to video_errors.csv...")
failed_df.to_csv('video_errors.csv', index=False, encoding='utf-8-sig')

# --- Final Summary ---
print("\nFinished processing.")
# Count how many entries in metadata_df do not have None title (as a proxy for success)
successful_count = metadata_df['title'].notna().sum()
print(f"Successfully fetched metadata for: {successful_count} videos (check video_full_metadata.csv).")
print(f"Failed to process or fetch full metadata for: {len(failed_df)} URLs (check video_errors.csv for details).")