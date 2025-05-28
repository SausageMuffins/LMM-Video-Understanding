#!/usr/bin/env python3

"""
Script to slow down videos and adjust their corresponding SRT captions.

This script reads video metadata from a CSV file, processes each video
using ffmpeg to slow it down by a specified factor, adjusts the timestamps
in the SRT captions accordingly, and saves the results along with updated
metadata.
"""

import os
import re
import pandas as pd
import subprocess
from tqdm import tqdm
import csv  # Imported in the notebook, kept for consistency though not directly used
import shutil  # Imported in the notebook, kept for consistency though not directly used
import argparse  # Added for potential future command-line arguments, but not strictly required by the prompt


def parse_arguments():
    """
    Parse command-line arguments for slowing videos.
    """
    parser = argparse.ArgumentParser(
        description="Slow down videos and adjust SRT captions."
    )
    parser.add_argument(
        "--input_metadata",
        "-m",
        type=str,
        required=True,
        help="Path to input metadata CSV. Must contain 'video_path' and 'caption' columns.",
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Base directory for input video files.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save slowed-down videos and updated metadata.",
    )
    parser.add_argument(
        "--speed_factor",
        "-s",
        type=float,
        default=0.5,
        help="Factor to slow down videos (e.g., 0.5 for half speed).",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        help="Process only this specific video ID",
    )
    return parser.parse_args()


# --- Function Definitions ---


def slow_down_video(input_path, output_path, speed_factor=0.5):
    """
    Slow down a video by the specified factor using ffmpeg.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        speed_factor (float, optional): Factor to slow down the video
                                        (e.g., 0.5 for half speed).
                                        Defaults to 0.5.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build ffmpeg command: use setpts filter to slow down video, atempo to slow down audio
    # atempo filter only supports 0.5-2.0 range, so for extreme slowing, we might need chaining (though not implemented here for simplicity based on original code)
    # Note: The original notebook command might fail if speed_factor is outside 0.5-2.0 for audio.
    # The provided code uses a single atempo filter, assuming speed_factor is within the valid range or ffmpeg handles it gracefully.
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-filter_complex",
        f"[0:v]setpts={1/speed_factor:.6f}*PTS[v];[0:a]atempo={speed_factor:.6f}[a]",  # Use floating point format specifier for precision
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",  # Video codec
        "-preset",
        "medium",  # Encoding speed/quality trade-off
        "-crf",
        "23",  # Constant Rate Factor (quality, lower is better, 18-28 typical) - Added reasonable default
        "-c:a",
        "aac",  # Audio codec
        "-b:a",
        "128k",  # Audio bitrate - Added reasonable default
        output_path,
        "-y",  # Overwrite output file if it exists
    ]

    print(f"Running command: {' '.join(cmd)}")  # Optional: print the command being run
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )  # Capture output for potential debugging
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path} with ffmpeg.")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        raise  # Re-raise the exception to stop the script or be caught higher up


def adjust_srt_timestamps(srt_text, speed_factor=0.5):
    """
    Adjust SRT timestamps by the specified speed factor.

    Args:
        srt_text (str): SRT format text.
        speed_factor (float, optional): Factor to slow down the timestamps
                                        (e.g., 0.5 means timestamps divided by 0.5,
                                        making durations longer). Defaults to 0.5.

    Returns:
        str: Adjusted SRT text. Returns original text if input is None or not a string.
    """
    if not isinstance(srt_text, str):
        return srt_text  # Return input if it's not a string (e.g., NaN)

    # Define a regex pattern to match SRT timestamp lines
    # Handles potential spaces around '-->'
    pattern = r"(\d{1,2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2},\d{3})"

    def time_to_seconds(time_str):
        """Convert SRT timestamp (HH:MM:SS,ms) to seconds."""
        try:
            time_parts = time_str.replace(",", ".").split(":")
            if len(time_parts) == 3:
                h, m, s = map(float, time_parts)
                return h * 3600 + m * 60 + s
            else:
                # Handle cases like MM:SS,ms if necessary, though standard is HH:MM:SS,ms
                print(
                    f"Warning: Unexpected time format '{time_str}'. Assuming HH:MM:SS,ms."
                )
                return 0.0  # Or raise an error
        except ValueError:
            print(f"Error converting time string: {time_str}")
            return 0.0  # Or raise an error

    def seconds_to_time(seconds):
        """Convert seconds to SRT timestamp format (HH:MM:SS,ms)."""
        if seconds < 0:
            seconds = 0  # Ensure non-negative time
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s_total = seconds % 60
        s = int(s_total)
        ms = int((s_total - s) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def adjust_timestamp(match):
        """Adjust a pair of timestamps based on the speed factor."""
        start_time_str = match.group(1)
        end_time_str = match.group(2)

        # Convert to seconds
        start_seconds = time_to_seconds(start_time_str)
        end_seconds = time_to_seconds(end_time_str)

        # Adjust time by dividing by speed factor (e.g., 0.5 makes duration 2x longer)
        # Avoid division by zero
        if speed_factor == 0:
            print("Warning: speed_factor cannot be zero. Timestamps not adjusted.")
            adjusted_start_seconds = start_seconds
            adjusted_end_seconds = end_seconds
        else:
            adjusted_start_seconds = start_seconds / speed_factor
            adjusted_end_seconds = end_seconds / speed_factor

        # Convert back to timestamp format
        adjusted_start_time = seconds_to_time(adjusted_start_seconds)
        adjusted_end_time = seconds_to_time(adjusted_end_seconds)

        return f"{adjusted_start_time} --> {adjusted_end_time}"

    # Use regex to find and replace all timestamp lines
    try:
        adjusted_srt = re.sub(pattern, adjust_timestamp, srt_text)
        return adjusted_srt
    except Exception as e:
        print(f"Error adjusting SRT timestamps: {e}")
        return srt_text  # Return original text on error


# --- Main Execution Block ---


def main():
    """
    Main function to execute the video processing workflow.
    """
    # Parse arguments
    args = parse_arguments()
    input_metadata_file = args.input_metadata
    input_video_base_dir = args.input_dir
    output_video_base_dir = args.output_dir
    speed_factor = args.speed_factor

    # Save updated metadata in the same folder as the input metadata
    output_metadata_file = os.path.join(
        os.path.dirname(input_metadata_file), os.path.basename(input_metadata_file)
    )

    # --- Setup ---
    print(f"Creating output folder: {output_video_base_dir}")
    os.makedirs(output_video_base_dir, exist_ok=True)

    # --- Process Metadata File ---
    print(f"Reading metadata file: {input_metadata_file}...")
    try:
        df = pd.read_csv(input_metadata_file)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {input_metadata_file}")
        return  # Exit if metadata is missing
    except Exception as e:
        print(f"Error reading metadata file {input_metadata_file}: {e}")
        return
    
    if args.video_id:
        df = df[df["video_path"].str.contains(args.video_id)]

    # --- Prepare DataFrame for Processing ---
    # Create copies of columns to modify safely
    df["slowed_video_path"] = df["video_path"].copy()
    df["slowed_caption"] = df["caption"].copy()
    df["processing_status"] = "pending"  # Add a status column
    df["processing_error"] = None  # Add an error message column

    # --- Process Videos and Update Metadata ---
    print("Processing videos and updating metadata...")
    processed_count = 0
    error_count = 0

    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        raw_path = row["video_path"]
        # Resolve video_path relative to input_dir if file not found directly
        if os.path.exists(raw_path):
            video_path = raw_path
        else:
            # try using basename under input_video_base_dir
            candidate = os.path.join(input_video_base_dir, os.path.basename(raw_path))
            if os.path.exists(candidate):
                video_path = candidate
            else:
                video_path = raw_path  # leave as-is, will trigger 'not found' below

        # Basic validation of video path
        if not isinstance(video_path, str) or not video_path:
            print(
                f"Warning: Invalid video path at index {idx}: {video_path}. Skipping."
            )
            df.at[idx, "processing_status"] = "skipped"
            df.at[idx, "processing_error"] = "Invalid video path"
            continue

        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video not found at index {idx}: {video_path}. Skipping.")
            df.at[idx, "processing_status"] = "skipped"
            df.at[idx, "processing_error"] = "Video file not found"
            continue

        # Determine output path: preserve directory structure only if under input_dir
        try:
            relative_path = os.path.relpath(video_path, input_video_base_dir)
            # If computed relative path goes outside the base dir, treat as not contained
            if relative_path.startswith(os.pardir):
                raise ValueError
            output_path = os.path.join(output_video_base_dir, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except Exception:
            # Fallback: place slowed video directly in output_dir
            print(
                f"Warning: cannot map '{video_path}' under '{input_video_base_dir}'. Using flat output structure."
            )
            output_filename = os.path.basename(video_path)
            output_path = os.path.join(output_video_base_dir, output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Slow down the video
        try:
            print(f"\nProcessing [{idx+1}/{len(df)}]: {video_path} -> {output_path}")
            slow_down_video(video_path, output_path, speed_factor=speed_factor)

            # Update metadata upon successful processing
            df.at[idx, "slowed_video_path"] = output_path
            df.at[idx, "processing_status"] = "success"
            processed_count += 1

            # Update caption timestamps if caption data exists and is a string
            if pd.notna(row["caption"]) and isinstance(row["caption"], str):
                print(f"Adjusting captions for {video_path}...")
                adjusted_caption = adjust_srt_timestamps(
                    row["caption"], speed_factor=speed_factor
                )
                df.at[idx, "slowed_caption"] = adjusted_caption
            else:
                print(
                    f"No valid caption found for {video_path} or caption is not text."
                )
                df.at[idx, "slowed_caption"] = row[
                    "caption"
                ]  # Keep original if not string or NaN

        except Exception as e:
            error_message = f"Error processing {video_path}: {str(e)}"
            print(f"\n{error_message}")
            df.at[idx, "processing_status"] = "error"
            df.at[idx, "processing_error"] = error_message
            # Keep original paths/captions in the temporary columns on error
            df.at[idx, "slowed_video_path"] = row["video_path"]
            df.at[idx, "slowed_caption"] = row["caption"]
            error_count += 1

    # --- Create Final Output DataFrame ---
    print("Creating final output dataframe...")
    output_df = df.copy()

    # Replace original columns with the processed data
    # Only update for successfully processed videos if desired, or update all (as per original notebook)
    # The original notebook replaced all, regardless of success, using the temp columns.
    output_df["video_path"] = df["slowed_video_path"]
    output_df["caption"] = df["slowed_caption"]

    # Adjust duration if the column exists
    if "duration" in output_df.columns:
        print("Adjusting duration column...")
        # Apply adjustment only where processing was successful? Or for all?
        # Original notebook did it for all. Let's stick to that.
        # Ensure duration is numeric before multiplying
        output_df["duration"] = pd.to_numeric(output_df["duration"], errors="coerce")
        if speed_factor != 0:
            output_df["duration"] = output_df["duration"] / speed_factor
        else:
            print("Warning: Speed factor is 0, duration not adjusted.")
    else:
        print(
            "Warning: 'duration' column not found in metadata. Skipping duration adjustment."
        )

    # Remove temporary and status columns before saving
    columns_to_drop = [
        "slowed_video_path",
        "slowed_caption",
        "processing_status",
        "processing_error",
    ]
    output_df = output_df.drop(
        columns=[col for col in columns_to_drop if col in output_df.columns], axis=1
    )

    # --- Save Updated Metadata ---
    print(f"Saving updated metadata to {output_metadata_file}...")
    try:
        output_df.to_csv(
            output_metadata_file, index=False, encoding="utf-8-sig"
        )  # Use utf-8-sig for better Excel compatibility
    except Exception as e:
        print(f"Error saving updated metadata file {output_metadata_file}: {e}")

    # --- Final Summary ---
    print("\n--- Processing Complete! ---")
    num_videos = len(df)
    # Recalculate processed based on status column for clarity
    actual_processed = df[df["processing_status"] == "success"].shape[0]
    skipped_count = df[df["processing_status"] == "skipped"].shape[0]

    print(f"Total entries in metadata: {num_videos}")
    print(f"Successfully processed:    {actual_processed}")
    print(f"Skipped (e.g., file missing): {skipped_count}")
    print(f"Encountered errors:      {error_count}")
    print(f"Slow videos saved under: {output_video_base_dir}")
    print(f"Updated metadata saved to: {output_metadata_file}")


if __name__ == "__main__":
    main()
