from prompts import divide_and_conquer_default_prompt, divide_and_conquer_default_prompt
from dotenv import load_dotenv  # Loads environment variables from a .env file
import google.generativeai as genai  # Gemini API library for video understanding
import os
import json
import time

# Load environment variables from .env and configure Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)


def upload_and_process_video(video_path):
    """
    Upload a video to Gemini API and wait for processing to complete.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        object or None: Processed video file object from Gemini API, or None if processing fails.
    """
    try:
        print(f"Uploading video: {video_path}")
        video_file = genai.upload_file(path=video_path)  # Upload the video file

        print("Waiting for video processing...")
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)  # Poll for updated status

        if video_file.state.name == "FAILED":
            raise ValueError(
                f"Video processing failed for {video_path}: {video_file.state.name}"
            )

        print("Video processed successfully.")
        return video_file
    except Exception as e:
        print(f"Error uploading/processing video {video_path}: {str(e)}")
        return None


def understand_video(video_path):
    """
    Perform video understanding using Gemini Flash for a given video file.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        str: The output text from Gemini API representing the video understanding.
    """
    video_file = upload_and_process_video(video_path)
    if video_file is None:
        return ""

    # Define a default prompt for video understanding
    prompt = (
        divide_and_conquer_default_prompt()
    )  # Default prompt for video understanding in prompts.py

    # Initialize the Gemini model for flash processing with a set temperature
    model = genai.GenerativeModel(
        model_name="models/gemini-2.0-flash", generation_config={"temperature": 0.5}
    )

    # Prepare input data for the model
    input_data = [prompt, video_file]

    print(f"Generating partial video understanding for {video_path} ...")
    response = model.generate_content(input_data, request_options={"timeout": 600})

    try:
        output = response.text.strip()
    except Exception as e:
        print(f"Error processing response for {video_path}: {str(e)}")
        output = ""

    return output


def process_videos(root_dir="videos", output_json="partial_understanding.json"):
    """
    Process video understanding for each video directory and save the outputs in a JSON file.

    Parameters:
        root_dir (str): Root directory containing subdirectories for each video.
        output_json (str): Name of the JSON file to store the outputs.
    """
    results = {}  # Dictionary to store outputs for each video ID

    # Loop through each subdirectory in the root directory
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue  # Skip if not a directory

        video_id = subdir  # Use the subdirectory name as the video ID
        print(f"\nProcessing video ID: {video_id}")
        results[video_id] = {}

        # Process each of the four mp4 files: {video_id}_0.mp4, {video_id}_1.mp4, {video_id}_2.mp4, {video_id}_3.mp4
        for i in range(4):
            filename = f"{video_id}_{i}.mp4"
            video_file_path = os.path.join(subdir_path, filename)

            if not os.path.exists(video_file_path):
                print(f"File not found: {video_file_path}")
                results[video_id][f"{video_id}_{i}"] = "File not found."
                continue

            print(f"Processing file: {filename}")
            output = understand_video(video_file_path)
            results[video_id][
                f"{video_id}_{i}"
            ] = output  # Record the output for this video segment

    # Save the results dictionary to a JSON file
    with open(output_json, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"\nResults saved to {output_json}")


if __name__ == "__main__":
    process_videos()
