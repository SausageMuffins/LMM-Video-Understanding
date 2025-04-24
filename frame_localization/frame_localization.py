import os
import json
import csv  # For reading CSV data
import cv2  # OpenCV for video frame extraction and image combination
import torch  # PyTorch for tensor and model operations
from transformers import CLIPProcessor, CLIPModel  # Hugging Face CLIP classes
import numpy as np
import csv

def extract_frames(video_path, every_n_frames=30):
    """
    Extract frames from a video file at regular intervals.
    
    Args:
        video_path (str): Path to the video file.
        every_n_frames (int): Interval between frames to extract.
        
    Returns:
        List of tuples (frame_index, frame image as a numpy array).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % every_n_frames == 0:
            frames.append((frame_index, frame))
        frame_index += 1
    cap.release()
    return frames

def get_top_k_frames_for_description(frames, description, processor, model, k=4):
    """
    Compute cosine similarity between a scene description and each video frame,
    then return the top k frames (in chronological order) with the highest scores.
    
    Args:
        frames (list): List of (frame_index, image) tuples.
        description (str): Scene description text.
        processor: CLIPProcessor from Hugging Face.
        model: CLIPModel from Hugging Face.
        k (int): Number of top frames to extract.
        
    Returns:
        List of (frame_index, image) tuples for the top k frames in chronological order.
    """
    print("Processing scene description:", description)
    inputs_text = processor(text=[description], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarities = []
    for idx, image in frames:
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs_image = processor(images=image_rgb, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs_image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
        similarities.append((idx, image, similarity))
    
    # Get top k by similarity, then sort those frames chronologically.
    top_k = sorted(similarities, key=lambda x: x[2], reverse=True)[:k]
    top_k_sorted = sorted(top_k, key=lambda x: x[0])
    return [(idx, image) for idx, image, sim in top_k_sorted]

def combine_frames_horizontally(frames):
    """
    Combine a list of frame images into a single image arranged horizontally.
    
    Args:
        frames (list): List of images as numpy arrays.
        
    Returns:
        Combined image as a numpy array.
    """
    return cv2.hconcat(frames)

def process_qid(qid, video_path, scene_data, processor, model, frame_interval=30, k=4):
    """
    Process one qid by generating a collage based solely on the 'generalized_scenes'.
    
    Args:
        qid (str): The query id.
        video_path (str): Path to the associated video file.
        scene_data (dict): Dictionary containing scene descriptions, including 'generalized_scenes'.
        processor: CLIPProcessor.
        model: CLIPModel.
        frame_interval (int): Interval for frame extraction.
        k (int): Number of top frames to extract per scene description.
    """
    output_dir = os.path.join("localized_frames", qid)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames from the video.
    frames = extract_frames(video_path, every_n_frames=frame_interval)
    if not frames:
        print(f"No frames extracted for qid {qid} from video {video_path}.")
        return
    
    # Use only the generalized scenes for collage generation.
    category = "generalized_scenes"
    scene_list = scene_data.get(category, [])
    for i, scene_desc in enumerate(scene_list):
        top_frames = get_top_k_frames_for_description(frames, scene_desc, processor, model, k=k)
        images = [img for idx, img in top_frames]
        if not images:
            print(f"No frames found for qid {qid} - {category} scene {i+1}. Skipping.")
            continue
        combined_image = combine_frames_horizontally(images) if len(images) > 1 else images[0]
        filename = f"{category}_{i+1}.jpg"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, combined_image)
        print(f"Saved collage for qid {qid} - {category} scene {i+1} at {out_path}")

if __name__ == "__main__":
    """
    Main function to load scene descriptions from a JSON file and the video mapping 
    from a CSV file, then process each qid to create localized frame collages based on generalized scenes.
    
    Assumptions:
    - The JSON file (json_descriptions.json) contains entries for each qid with keys:
      'overview', 'expected_scenes', 'unexpected_scenes', and 'generalized_scenes'.
    - The CSV file (challenge_data_corrected.csv) provides a mapping from qid to video_id.
      Video files are located in the "data" folder and named as {video_id}.mp4 after stripping extra quotes.
    """
    

    # Build video mapping and (optionally) scene descriptions from CSV.
    csv_file = "challenge_data_corrected.csv"  # Path to CSV file with video mapping.
    video_mapping = {}  # Mapping from qid to video file path.
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("\ufeffqid", row.get("qid"))
            video_id = row["video_id"].strip('"')
            video_path = os.path.join("data", f"{video_id}.mp4")
            video_mapping[qid] = video_path

    # Load scene descriptions from JSON.
    json_file = "json_descriptions.json"
    with open(json_file, "r", encoding="utf-8") as f:
        scene_descriptions = json.load(f)
    
    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    
    for qid, scene_data in scene_descriptions.items():
        video_path = video_mapping.get(qid)
        if video_path is None:
            print(f"No video mapping found for qid: {qid}. Skipping.")
            continue
        print(f"Processing qid: {qid} with video: {video_path}")
        process_qid(qid, video_path, scene_data, processor, model, frame_interval=30, k=7)
