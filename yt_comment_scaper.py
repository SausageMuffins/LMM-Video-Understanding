from googleapiclient.discovery import build
import pandas as pd
import os
import re
import argparse
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Scrape YouTube comments from videos listed in a CSV file.')
    parser.add_argument('--input', '-i', type=str, default='video_full_metadata.csv',
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
    if not youtube_url or not isinstance(youtube_url, str):
        return None
        
    parsed_url = urlparse(youtube_url)
    
    # Handle youtu.be domain
    if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path[1:]
        
    # Handle youtube.com domain
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
        # Regular watch URLs
        if parsed_url.path == '/watch':
            query = parse_qs(parsed_url.query)
            if 'v' in query:
                return query['v'][0]
        
        # Shorts URLs format: /shorts/VIDEO_ID
        elif parsed_url.path.startswith('/shorts/'):
            # Extract the video ID from the path
            path_parts = parsed_url.path.split('/')
            if len(path_parts) >= 3:  # Should be ['', 'shorts', 'VIDEO_ID', ...]
                return path_parts[2]
        
        # Embed URLs
        elif parsed_url.path.startswith('/embed/'):
            path_parts = parsed_url.path.split('/')
            if len(path_parts) >= 3:
                return path_parts[2]
        
        # Old-style URLs
        elif parsed_url.path.startswith('/v/'):
            path_parts = parsed_url.path.split('/')
            if len(path_parts) >= 3:
                return path_parts[2]
    
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
                snippet = item['snippet']['topLevelComment']['snippet']
                comment = snippet['textDisplay']
                author = snippet['authorDisplayName']
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
                    
            if items_processed < max_results and 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=min(100, max_results - items_processed)
                )
                response = request.execute()
            else:
                break
                
    except Exception as e:
        print(f"Error getting comments for video {video_id}: {str(e)}")
    
    return comments[:max_results]  # Ensure we don't return more than max_results

def sanitize_filename(title):
    """Convert a title to a valid filename."""
    # Remove invalid filename characters
    clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
    # Replace spaces with underscores and limit length
    clean_title = clean_title.replace(' ', '_')[:100]
    return clean_title

def main():
    """Main function to scrape comments based on URLs in the input CSV."""
    # Parse command line arguments
    args = parse_arguments()
    input_csv = args.input
    output_dir = args.output
    max_comments = args.comments
    verbose = args.verbose
    
    # Initialize YouTube API
    youtube = build("youtube", "v3", developerKey=api_key)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    
    try:
        # Read the input CSV
        if verbose:
            print(f"Reading input CSV: {input_csv}")
        df = pd.read_csv(input_csv)
        
        # Check if required columns exist
        if 'youtube_url' not in df.columns:
            raise ValueError("Input CSV must contain a 'youtube_url' column")
        if 'title' not in df.columns:
            raise ValueError("Input CSV must contain a 'title' column")
        
        # Process each video
        total_videos = len(df)
        for index, row in df.iterrows():
            youtube_url = row['youtube_url']
            title = row['title']
            
            if verbose:
                print(f"\nProcessing video {index+1}/{total_videos}: {title}")
            else:
                print(f"Processing video {index+1}/{total_videos}: {title}")
            
            # Clean the title to use as a filename
            clean_title = sanitize_filename(title)
            
            # Extract video ID
            video_id = extract_video_id(youtube_url)
            if not video_id:
                print(f"  Could not extract video ID from URL: {youtube_url}")
                continue
            
            if verbose:
                print(f"  Video ID: {video_id}")
                print(f"  Scraping up to {max_comments} comments...")
            
            # Get comments
            comments = get_comments(youtube, video_id, max_results=max_comments)
            
            if comments:
                # Create a DataFrame and save to CSV
                output_file = os.path.join(output_dir, f"{clean_title}.csv")
                comments_df = pd.DataFrame(comments)
                comments_df.to_csv(output_file, index=False)
                print(f"  Saved {len(comments)} comments to {output_file}")
            else:
                print(f"  No comments found for video: {title}")
    
    except FileNotFoundError:
        print(f"Error: Input CSV file '{input_csv}' not found.")
    except Exception as e:
        print(f"Error processing the input CSV: {str(e)}")

if __name__ == "__main__":
    main()