"""
This script loads a video understanding model and processor.
It iterates through all MP4 files in a given directory,
applies the chat template to generate inputs, and prints the decoded response.
"""

import os
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# Initialize the model in half-precision and load the processor
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    torch_dtype=torch.float16,
    device_map="auto",  # Load model with half-precision
)

model.eval()  # Set the model to evaluation mode
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
# processor.tokenizer.padding_side = "left"  # advice from hugging-face

# Directory containing MP4 videos
video_directory = "videos"  # Update this path to your directory

# Get the device on which the model is loaded (e.g., 'mps' or 'cuda')
device = model.device

# Loop through each MP4 file in the directory
for filename in os.listdir(video_directory):
    if filename.lower().endswith(".mp4"):
        video_path = os.path.join(video_directory, filename)
        # print("\n" + "=" * 80)
        # print(f"Processing video: {video_path}\n")

        # Build the conversation for the current video
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the color of the jacket of the guy?",
                    },
                    {"type": "video", "path": video_path},
                ],
            },
        ]

        # Create inputs using the processor; it uniformly samples 8 frames from the video
        inputs = processor.apply_chat_template(
            conversation,
            num_frames=32,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        # Generate the response text with additional generation parameters
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,  # Use sampling to generate text
            temperature=0.6,  # adjust temperature to control randomness
            # num_beams=5,  # Use beam search with 5 candidate sequences
            # early_stopping=False,  # Stop when the best sequence is complete
        )

        # Debug: print raw output tokens for additional insight
        print("Raw output tokens:", out)
        print(len(out))

        # Decode the generated output into a human-readable string
        result = processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # print(result)
        # Print the decoded result in a clearer format
        print(f"Decoded Result for '{filename}':\n{result}")
        print("=" * 80)

        # Remove the break to process all videos in the directory
        break  # Remove or comment this line to process all videos in the directory
