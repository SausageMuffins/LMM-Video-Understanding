import os
import av
import torch
import numpy as np
import pandas as pd
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

# ----------------------------------------------------------------------------
# 1. Read CSV Files as Copies
# ----------------------------------------------------------------------------
CHALLENGE_DATA_PATH = "challenge_data.csv"
VIDEO_METADATA_PATH = "video_metadata.csv"
OUTPUT_CSV_PATH = "challenge_data_answers.csv"

df_challenge_data = pd.read_csv(
    CHALLENGE_DATA_PATH
).copy()  # Copy to avoid altering original
df_video_metadata = pd.read_csv(
    VIDEO_METADATA_PATH
).copy()  # Copy to avoid altering original

# Ensure the "answer" column can hold string data (if it doesn't exist, create it)
if "answer" not in df_challenge_data.columns:
    df_challenge_data["answer"] = ""

df_challenge_data["answer"] = df_challenge_data["answer"].astype(str)

# ----------------------------------------------------------------------------
# 2. Load the Model and Processor
# ----------------------------------------------------------------------------
model_id = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)
model.eval()

processor = LlavaNextVideoProcessor.from_pretrained(model_id)


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container for reading the video.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: Array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# ----------------------------------------------------------------------------
# 3. Loop Through Each Entry in Challenge Data
# ----------------------------------------------------------------------------
for idx, row in df_challenge_data.iterrows():
    youtube_url = row["youtube_url"]

    # Combine question and question_prompt into a single prompt
    combined_prompt_text = f"{row['question']} {row['question_prompt']}"

    # Prepare the conversation structure for LLaVA
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": combined_prompt_text},
                {"type": "video"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # ----------------------------------------------------------------------------
    # 4. Find the Matching Video Path from Video Metadata
    # ----------------------------------------------------------------------------
    matching_metadata = df_video_metadata[
        df_video_metadata["youtube_url"] == youtube_url
    ]
    if matching_metadata.empty:
        print(f"No metadata found for youtube_url: {youtube_url}")
        continue

    video_path = matching_metadata.iloc[0]["video_path"]
    if not os.path.exists(video_path):
        print(f"Video file not found at path: {video_path}")
        continue

    # ----------------------------------------------------------------------------
    # 5. Process the Video and Get the Model Output
    # ----------------------------------------------------------------------------
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames < 1:
        print(f"Video {video_path} has no frames. Skipping.")
        continue

    # Sample uniformly 8 frames (or fewer if the video has fewer frames)
    frames_to_sample = min(total_frames, 8)
    indices = np.arange(0, total_frames, total_frames / frames_to_sample).astype(int)

    clip = read_video_pyav(container, indices)
    inputs_video = processor(
        text=prompt, videos=clip, padding=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=256, do_sample=False)
    decoded_output = processor.decode(output[0][2:], skip_special_tokens=True)

    # ----------------------------------------------------------------------------
    # 6. Process Model Outputs
    # ----------------------------------------------------------------------------
    assistant_marker = "ASSISTANT: "
    assistant_index = decoded_output.find(assistant_marker)
    if assistant_index != -1:
        # Keep only the text that appears after "ASSISTANT: "
        final_answer = decoded_output[assistant_index + len(assistant_marker) :].strip()
    else:
        final_answer = decoded_output.strip()

    # Store the truncated answer in the DataFrame
    df_challenge_data.at[idx, "answer"] = final_answer

    print(f"Processed QID={row['qid']} with youtube_url={youtube_url}")
    print(f"Answer: {final_answer}")
    print("-" * 50)

# ----------------------------------------------------------------------------
# 7. Write the Updated DataFrame to a New CSV
# ----------------------------------------------------------------------------
df_challenge_data.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Answers saved to: {OUTPUT_CSV_PATH}")
