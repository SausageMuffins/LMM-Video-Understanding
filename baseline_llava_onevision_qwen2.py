"""
This script reads challenge and video metadata CSVs, processes each video by sampling frames,
and uses the llava-onevision-qwen2-7b-ov-chat model to generate an answer.
It loads the model via the official builder so that the model can properly handle video (multi-frame) inputs.

It is required to install the following package before running this script:
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

To use flash attention, you can install the following package:
pip install flash-attn --no-build-isolation

(ryan: but this requires an nvidia gpu :< im a very sad macbook user)

current implementation does not use flash attention 2
"""

import os
import av
import torch
import numpy as np
import pandas as pd
import copy
import warnings
from PIL import Image  # For image conversion

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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

# Ensure the "answer" column exists and can hold string data
if "answer" not in df_challenge_data.columns:
    df_challenge_data["answer"] = ""
df_challenge_data["answer"] = df_challenge_data["answer"].astype(str)

# ----------------------------------------------------------------------------
# 2. Load the Model and Utilities (Following Documentation)
# ----------------------------------------------------------------------------
# Import required functions and constants from the LLaVA repository
from llava.model.builder import (
    load_pretrained_model,
)  # Loads model, tokenizer, and image processor
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)  # For processing images and tokenization
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)  # Constants for image tokens
from llava.conversation import conv_templates  # Predefined conversation templates

# Define model parameters
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
model_name = "llava_qwen"
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

device_map = "auto"

# Load tokenizer, model, image_processor, and max_length using the official builder.
# We set attn_implementation="eager" (as an alternative to flash attention)
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained,
    None,
    model_name,
    device_map=device_map,
    attn_implementation="eager",  # Use manual attention implementation
)
model.to(device)
model.eval()


# ----------------------------------------------------------------------------
# 3. Utility Function: Decode Video Frames with PyAV
# ----------------------------------------------------------------------------
def read_video_pyav(container, indices):
    """
    Decode video frames using the PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container for reading the video.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: Array of decoded frames with shape (num_frames, height, width, 3).
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
# 4. Loop Through Each Entry in Challenge Data and Process Videos
# ----------------------------------------------------------------------------
for idx, row in df_challenge_data.iterrows():
    youtube_url = row["youtube_url"]
    combined_prompt_text = f"{row['question']} {row['question_prompt']}"

    # 4a. Prepare the Conversation Prompt Using the Template
    # Use the qwen_1_5 conversation template (ensure you use the correct one for qwen models)
    conv_template = "qwen_1_5"
    # Prepend the default image token (required by the model) to the prompt text
    question = DEFAULT_IMAGE_TOKEN + "\n" + combined_prompt_text
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # 4b. Find the Matching Video Path from Video Metadata
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

    # 4c. Process the Video: Sample Frames and Convert to List of PIL Images
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames < 1:
        print(f"Video {video_path} has no frames. Skipping.")
        continue

    # Uniformly sample up to 8 frames
    frames_to_sample = min(total_frames, 8)
    indices = np.linspace(0, total_frames - 1, num=frames_to_sample, dtype=int)
    clip = read_video_pyav(container, indices)
    # Convert each frame (numpy array) to a PIL image
    pil_images = [Image.fromarray(frame) for frame in clip]
    # Record the sizes of each image (resolution)
    image_sizes = [img.size for img in pil_images]

    # Process the list of PIL images into image tensors using the image_processor
    image_tensor = process_images(pil_images, image_processor, model.config)
    # Move each processed tensor to the correct device and set the data type
    image_tensor = [
        _image.to(dtype=torch.float16, device=device) for _image in image_tensor
    ]

    # 4d. Tokenize the Prompt and Generate the Model Output
    # Tokenize the prompt to get input_ids (using tokenizer_image_token from LLaVA)
    input_ids = (
        tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device)
    )  # shape: (1, seq_len)

    # Generate output without passing an attention_mask (following documentation code)
    output = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,  # Using higher max tokens as in documentation
    )
    text_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
    # In the documentation, the output is printed directly without additional processing
    final_answer = text_outputs[0] if isinstance(text_outputs, list) else text_outputs

    # 4e. Process the Generated Output (directly use the decoded text)
    # No need to search for a marker; we use the output as-is
    df_challenge_data.at[idx, "answer"] = final_answer

    print(f"Processed QID={row['qid']} with youtube_url={youtube_url}")
    print(f"Answer: {final_answer}")
    print("-" * 50)

# ----------------------------------------------------------------------------
# 5. Write the Updated DataFrame to a New CSV
# ----------------------------------------------------------------------------
df_challenge_data.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Answers saved to: {OUTPUT_CSV_PATH}")
