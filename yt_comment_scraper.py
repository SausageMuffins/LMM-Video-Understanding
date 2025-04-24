#!/usr/bin/env python3

from googleapiclient.discovery import build
import pandas as pd
import os
import re
import argparse
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import math # Import math to check for nan

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Scrape YouTube comments from videos listed in a CSV file.')
    parser.add_argument('--input', '-i', type=str, default='video_metadata.csv',
                        help='Path to input CSV file (default: video_metadata.csv)')
    parser.add_argument('--output', '-o', type=str, default='comments',
                        help='Directory to save comment CSV files (default: comments)')
    parser.add_argument('--comments', '-c', type=int, default=50,
                        help='Number of comments to scrape per video (default: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv('YOUTUBE_API_KEY')
if not api_key:
    raise ValueError("YOUTUBE_API_KEY not found in .env file")

def extract_video_id(youtube_url):
    """Extract the video ID from a YouTube URL, including Shorts URLs."""
    # Check if input is a valid non-empty string
    if not isinstance(youtube_url, str) or not youtube_url.strip():
        return None

    try:
        parsed_url = urlparse(youtube_url)

        # Handle youtu.be domain
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            video_id = parsed_url.path[1:]
            # Remove query params from youtu.be links if any
            if '?' in video_id:
                video_id = video_id.split('?')[0]
            return video_id if video_id else None # Check if path is empty

        # Handle youtube.com domain
        if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            # Regular watch URLs
            if parsed_url.path == '/watch':
                query = parse_qs(parsed_url.query)
                if 'v' in query and query['v'][0]:
                    return query['v'][0]

            # Shorts URLs format: /shorts/VIDEO_ID
            elif parsed_url.path.startswith('/shorts/'):
                path_parts = parsed_url.path.split('/')
                if len(path_parts) >= 3 and path_parts[2]:
                    return path_parts[2]

            # Embed URLs
            elif parsed_url.path.startswith('/embed/'):
                path_parts = parsed_url.path.split('/')
                if len(path_parts) >= 3 and path_parts[2]:
                    return path_parts[2]

            # Old-style URLs
            elif parsed_url.path.startswith('/v/'):
                path_parts = parsed_url.path.split('/')
                if len(path_parts) >= 3 and path_parts[2]:
                    return path_parts[2]
    except Exception as e:
        # Catch any parsing errors
        print(f"  Error parsing URL {youtube_url}: {e}")
        return None

    # Could not extract ID
    return None


def get_comments(youtube, video_id, max_results=50):
    """Get comments for a given video ID."""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results),  # API maximum is 100 per request
            order="relevance",  # Can be 'time', 'relevance', 'rating', 'video'
        )
        response = request.execute()

        items_processed = 0

        while response and items_processed < max_results:
            for item in response.get('items', []):
                # Check if the necessary nested structure exists
                if 'snippet' in item and 'topLevelComment' in item['snippet'] and \
                   'snippet' in item['snippet']['topLevelComment']:
                    snippet = item['snippet']['topLevelComment']['snippet']
                    comment = snippet.get('textDisplay', '') # Use .get for safety
                    author = snippet.get('authorDisplayName', 'Unknown Author')
                    like_count = snippet.get('likeCount', 0)
                    published_at = snippet.get('publishedAt', '')

                    comments.append({
                        "Author": author,
                        "Comment": comment,
                        "LikeCount": like_count,
                        "PublishedAt": published_at
                    })

                    items_processed += 1
                    if items_processed >= max_results:
                        break
                else:
                     print(f"  Skipping malformed comment item for video {video_id}")


            # Check for next page token only if we haven't reached max results
            if items_processed < max_results and 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=min(100, max_results - items_processed),
                    order="relevance" # Keep order consistent
                )
                response = request.execute()
            else:
                break # Exit loop if max results reached or no next page

    # Handle potential API errors gracefully (e.g., comments disabled)
    except Exception as e:
        # Check for common API error messages
        error_message = str(e)
        if 'disabled comments' in error_message.lower():
            print(f"  Comments are disabled for video {video_id}.")
        elif 'forbidden' in error_message.lower() or 'checkCarter' in error_message:
             print(f"  Access forbidden or restricted for comments on video {video_id}.")
        else:
            print(f"  Error getting comments for video {video_id}: {error_message}")

    return comments[:max_results]  # Ensure we don't return more than max_results

def sanitize_filename(title):
    """Convert a title to a valid filename."""
    # --- Added Check: Ensure input is a string ---
    if not isinstance(title, str):
        # If not a string (like NaN), return a default or perhaps raise error
        # For now, let's return an empty string which the calling code handles
        print(f"  Warning: sanitize_filename received non-string input: {type(title)}. Using default.")
        return "invalid_title" # Or return None / raise ValueError

    # Remove invalid filename characters
    # Replace sequences of whitespace with single space, then strip leading/trailing
    clean_title = re.sub(r'\s+', ' ', title).strip()
    # Remove invalid filename characters
    clean_title = re.sub(r'[\\/*?:"<>|]', "", clean_title)
    # Replace spaces with underscores and limit length
    clean_title = clean_title.replace(' ', '_')[:100] # Limit length AFTER cleaning
    # Handle cases where title becomes empty after cleaning
    if not clean_title:
        return "empty_title"
    return clean_title

def main():
    """Main function to scrape comments based on URLs in the input CSV."""
    args = parse_arguments()
    input_csv = args.input
    output_dir = args.output
    max_comments = args.comments
    verbose = args.verbose

    youtube = build("youtube", "v3", developerKey=api_key)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")

    try:
        if verbose:
            print(f"Reading input CSV: {input_csv}")
        df = pd.read_csv(input_csv)

        required_cols = ['youtube_url', 'title']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing_cols)}")

        total_videos = len(df)
        print(f"Found {total_videos} rows to process.")

        # --- Wrap the loop iteration in a try-except block to allow continuation ---
        for index, row in df.iterrows():
            try: # Start of per-row error handling
                youtube_url = row['youtube_url']
                title = row['title']

                # --- Check for NaN/invalid URL and Title ---
                if pd.isna(youtube_url) or not isinstance(youtube_url, str) or not youtube_url.strip():
                    print(f"\nSkipping row {index+1}/{total_videos}: Invalid or missing YouTube URL.")
                    continue # Skip to the next row

                # Handle NaN titles specifically for printing/logging
                display_title = title if isinstance(title, str) else "[Missing Title]"

                # Decide how to print based on verbosity
                print_prefix = f"\nProcessing video {index+1}/{total_videos}:"
                if verbose:
                    print(f"{print_prefix} {display_title} ({youtube_url})")
                else:
                    # Only print title if it's valid for non-verbose mode
                    if isinstance(title, str):
                         print(f"{print_prefix} {title}")
                    else:
                         print(f"{print_prefix} [Processing URL: {youtube_url}]")


                # --- Extract video ID (already handles non-string URL) ---
                video_id = extract_video_id(youtube_url)
                if not video_id:
                    print(f"  Could not extract valid video ID from URL: {youtube_url}. Skipping.")
                    continue # Skip to the next row if ID extraction fails

                if verbose:
                    print(f"  Video ID: {video_id}")
                    print(f"  Scraping up to {max_comments} comments...")

                # --- Get comments ---
                comments = get_comments(youtube, video_id, max_results=max_comments)

                if comments:
                    # --- Save comments using video_id as filename ---
                    output_file = os.path.join(output_dir, f"{video_id}.csv")
                    try:
                        comments_df = pd.DataFrame(comments)
                        comments_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                        print(f"  Saved {len(comments)} comments to {output_file}")
                    except Exception as e_csv:
                        print(f"  Error saving comments CSV for video ID {video_id}: {e_csv}")
                elif verbose or not 'disabled comments' in str(e).lower(): # Only print 'no comments' if not known to be disabled
                    print(f"  No comments found or retrieved for video: {display_title} (ID: {video_id})")

            except Exception as e_row: # Catch errors specific to this row
                print(f"\n--- Error processing row {index+1} (URL: {row.get('youtube_url', 'N/A')}) ---")
                print(f"  Error details: {str(e_row)}")
                print(f"  Skipping to next row.")
                # Optionally log this error to a separate file
                continue # Ensure loop continues

    except FileNotFoundError:
        print(f"Error: Input CSV file '{input_csv}' not found.")
    except ValueError as ve: # Catch specific errors like missing columns
         print(f"Configuration Error: {str(ve)}")
    except Exception as e: # Catch broader errors during setup/file reading
        print(f"An unexpected error occurred: {str(e)}")

    print("\nComment scraping process finished.")

if __name__ == "__main__":
    main()