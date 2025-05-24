#!/bin/bash

# Shell script to run the complete Gemini inference pipeline sequentially

set -e  # Exit on error

echo "[Step 1] Scraping YouTube Comments..."
python yt_comment_scraper.py \
    --input video_full_metadata.csv \
    --output comments \
    --comments 50 \
    --verbose

echo "[Step 2] Creating slowed-down videos..."
python create_slower_vids.py \
    --input_metadata video_full_metadata.csv \
    --input_dir videos_full \
    --output_dir videos_full_slow

echo "[Step 3] Running Gemini inference..."
python gemini.py \
    --challenge_data_path challenge_data.csv \
    --video_metadata_path videos_full_slow_metadata.csv \
    --comments_dir comments \
    --output_csv_path results/challenge_data_gemini_pro_2.5_exp_slowVid_v3_full.csv \
    --model_name models/gemini-2.5-pro-exp-03-25 \
    --v3 \
    --temperature 1.0
    # Uncomment below if needed:
    # --rate_limit 2

echo "[Step 4] Converting to submission format..."
python convert_to_upload_format.py \
    --input results/challenge_data_gemini_pro_2.5_exp_slowVid_v3_full.csv \
    --output submissions/pred_gemini_pro_2.5_exp_slowVid_v3_full.csv

echo "Pipeline execution completed successfully."