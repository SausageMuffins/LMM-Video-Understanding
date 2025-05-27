#!/bin/bash

# Demo pipeline: download metadata, scrape comments, slow videos,
# run inference, and format submission — all under demo/

set -e  # exit on any command failure

echo "[Step 0] Downloading challenge metadata into demo/…"
# Ensure demo folder exists
mkdir -p demo

# Run the metadata script from the parent directory, writing into demo/
pushd demo >/dev/null
python ../demo_download_vids_metadata.py
popd >/dev/null

# Prepare the rest of the demo subdirectories
mkdir -p demo/comments \
         demo/videos_full_slow \
         demo/results \
         demo/submissions

echo "[Step 1] Scraping YouTube Comments…"
python yt_comment_scraper.py \
  --input demo/video_full_metadata.csv \
  --output demo/comments \
  --comments 50 \
  --verbose

echo "[Step 2] Creating slowed-down videos…"
python demo_create_slower_vids.py \
  --input_metadata demo/video_full_metadata.csv \
  --input_dir demo/videos_full \
  --output_dir demo/videos_full_slow

echo "[Step 3] Running Gemini inference…"
python gemini.py \
  --challenge_data_path demo/challenge_data.csv \
  --video_metadata_path demo/video_full_metadata.csv \
  --comments_dir demo/comments \
  --output_csv_path demo/results/demo_gemini_output.csv \
  --model_name models/gemini-2.5-pro-exp-03-25 \
  --v3 \
  --temperature 1.0
# #  --rate_limit 2   # uncomment if you need to throttle

# echo "[Step 4] Converting to submission format…"
# python convert_to_upload_format.py \
#   --input demo/results/demo_gemini_output.csv \
#   --output demo/submissions/demo_submission.csv

echo "Demo pipeline execution completed successfully."