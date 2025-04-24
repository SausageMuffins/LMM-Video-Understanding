"""
Script to move a model file from the Hugging Face cache to a new directory.

This script:
  1. Defines the source file path in your Hugging Face cache.
  2. Defines the destination directory (/mnt/sda/ryan) and builds the destination path.
  3. Copies the file using shutil.copy2 (which preserves metadata).
  4. Deletes the original file from the cache.

Usage:
  python move_model.py
"""

import os
import shutil
import sys

def move_model(source, destination_dir):
    # Expand source if it uses ~
    source = os.path.expanduser(source)
    
    # Check if the source file exists
    if not os.path.exists(source):
        print(f"Source file not found: {source}", file=sys.stderr)
        sys.exit(1)
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Build the full destination file path
    filename = os.path.basename(source)
    destination = os.path.join(destination_dir, filename)
    
    try:
        # Copy the file (preserving metadata)
        shutil.copy2(source, destination)
        print(f"File copied to: {destination}")
        
        # Delete the original file
        os.remove(source)
        print(f"Original file deleted from: {source}")
    except Exception as e:
        print(f"Error during move operation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Define the source model file (adjust if needed)
    source_model = "~/.cache/huggingface/hub/models--bartowski--DeepSeek-R1-Distill-Qwen-32B-GGUF/snapshots/1dc8cf9ffa5dd333057ea1b09ccf4772d8726dec/DeepSeek-R1-Distill-Qwen-32B-Q6_K.gguf"
    # Define the destination directory
    destination_dir = "/mnt/sda/ryan"
    
    move_model(source_model, destination_dir)