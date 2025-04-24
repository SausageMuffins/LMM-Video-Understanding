# A&T: Augment and Think Before Answering
National AI Student Challenge 2025 - TikTok Track

![Solution Architecture](./images/Tiktok%20Solution%20Architecture.png)

[Slide Link](https://docs.google.com/presentation/d/1azfFHL0cmd4gwEwBs1MidxDgfFCrGLPxAPBd6n1hXmM/edit?usp=sharing)

## Our Solution Overview

Our approach focuses on **enhancing the context provided to the MLLM** and **pre-processing the visual input** to better align with the model's processing characteristics. We believe that by providing richer, more relevant information and optimising the video format, the MLLM can make more robust and consistent inferences.

**Core Ideas:**

1.  **Rich Context Augmentation:** We gather extensive metadata associated with each YouTube video, including title, description, channel information, keywords, **captions (transcripts)**, and critically, **user comments**. Comments often contain discussions, clarifications, or reveal audience perception, which can be vital clues (e.g., users pointing out a video is reversed).
2.  **Video Pre-processing (Slowing Down):** Recognising that models like Google Gemini process video at a fixed rate (e.g., 1 frame per second), we slow down the input videos (to 0.5x speed). Our intuition is that this allows the model to observe finer-grained temporal details within its fixed sampling rate, potentially making subtle motion patterns (like unnatural reversals) more apparent.
3.  **Structured Prompt Engineering:** We use Google's Gemini Pro 2.5 model with a carefully crafted prompt (`v3`). This prompt explicitly asks the model to:
    *   Describe the video's start, middle, and end segments.
    *   Detail its reasoning process step-by-step (`THINKING STEPS`).
    *   Provide a final answer (`FINAL ANSWER`).
    This structured approach encourages more thorough analysis before committing to an answer and leverages the rich context gathered in the previous steps.

## Solution Architecture

Our solution follows a multi-stage pipeline:

1.  **Data Ingestion & Initial Metadata:** Download the benchmark dataset (`challenge_data.csv`) from Hugging Face. Use `pytubefix` via the `download_vids_metadata.py` script to retrieve basic metadata (title, description, duration, captions) for each video URL and save it to `video_full_metadata.csv`.
2.  **Comment Scraping:** Use the YouTube Data API v3 via `yt_comment_scraper.py` to fetch relevant user comments for each video, adding another layer of contextual information.
3.  **Video Pre-processing:** Employ `ffmpeg` via `create_slower_vids.py` to slow down each video to half speed, adjusting caption timestamps accordingly. This generates the input videos used for inference and saves updated metadata to `videos_full_slow_metadata.csv`.
4.  **Context Aggregation & Prompting:** For each question in the benchmark (`challenge_data.csv`), the `gemini.py` script loads the corresponding slowed video and its associated metadata (title, description, channel, keywords, adjusted captions, comments). It then constructs the detailed `v3` prompt, integrating all context.
5.  **MLLM Inference:** The `gemini.py` script sends the slowed video and the rich prompt to the Google Gemini Pro 2.5 API.
6.  **Response Parsing & Formatting:** `gemini.py` parses the structured response from Gemini (start/middle/end descriptions, thinking steps, final answer). The `convert_to_upload_format.py` script then formats the final answer according to the submission requirements.

*(A dedicated architecture slide is included in our submission package as required.)*

## Features & Functionality

*   **Comprehensive Context:** Leverages not just video content but also titles, descriptions, captions, channel info, and user comments for holistic understanding.
*   **Optimised Video Input:** Slows down videos to potentially enhance detail perception by the MLLM's fixed-rate frame sampling.
*   **Advanced Prompting:** Uses a structured, multi-part prompt (`v3`) with Gemini Pro 2.5 to guide the model towards more reasoned and detailed analysis.
*   **Reproducibility:** The entire pipeline is scripted for easy reproduction of results.
*   **Modular Design:** Data collection, pre-processing, and inference are separated into distinct scripts.

## Tech Stack & Requirements

*   **Programming Language:** Python 3.x
*   **Core Libraries:**
    *   `pandas`: Data manipulation and CSV handling.
    *   `datasets` (from Hugging Face): To load the initial challenge data.
    *   `google-generativeai`: Interacting with the Gemini API.
    *   `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`: Interacting with the YouTube Data API v3.
    *   `python-dotenv`: Managing API keys securely.
    *   `pytubefix`: Downloading YouTube videos and metadata.
    *   `tqdm`: Progress bars for long processes.
    *   `requests` (likely a dependency of other libraries)
    *   `argparse`: Command-line argument parsing.
*   **External Tools:**
    *   `ffmpeg`: Required for video processing (slowing down). Must be installed separately and accessible in the system's PATH.
*   **APIs Used:**
    *   Google Gemini API (specifically `gemini-2.5-pro-exp-03-25`)
    *   YouTube Data API v3
*   **Models Used:**
    *   `gemini-2.5-pro-exp-03-25` (via Google AI Studio / API)
*   **Assets & Datasets:**
    *   `lmms-lab/AISG_Challenge` (Hugging Face Dataset): Contains the benchmark video URLs and questions. The script `download_vids_metadata.py` saves the relevant part as `challenge_data.csv`.
    *   YouTube Videos: Assumed to be pre-downloaded into the `videos_full/` directory based on the benchmark dataset URLs.
    *   Derived Metadata (`video_full_metadata.csv`, `videos_full_slow_metadata.csv`): Generated during the pipeline.
    *   Derived Comments (`comments/*.csv`): Generated during the pipeline.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Directory]
    ```

2.  **Install FFmpeg:**
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Windows:** Download from the official FFmpeg website and add it to your system's PATH.
    *   Verify installation: `ffmpeg -version`

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Python Dependencies:**
    *(Ensure you have created a `requirements.txt` file containing all necessary libraries listed in the Tech Stack section)*
    ```bash
    pip install -r requirements.txt
    ```
    *(If no `requirements.txt` is provided, manually install: `pip install pandas datasets google-generativeai google-api-python-client google-auth-oauthlib google-auth-httplib2 python-dotenv pytubefix tqdm requests argparse`)*

5.  **Set up API Keys:**
    Create a file named `.env` in the root project directory with the following content:
    ```env
    YOUTUBE_API_KEY=your_youtube_data_api_v3_key
    GEMINI_API_KEY=your_google_gemini_api_key
    ```
    Replace the placeholders with your actual API keys. You can obtain these from Google Cloud Console and Google AI Studio, respectively. Ensure you have enabled the necessary APIs (YouTube Data API v3, Generative Language API).

6.  **Prepare Benchmark Videos:**
    *   Download the video files corresponding to the URLs in the `lmms-lab/AISG_Challenge` dataset. The easiest way might be to download the provided zip: `https://huggingface.co/datasets/lmms-lab/AISG_Challenge/resolve/main/Benchmark-AllVideos-HQ-Encoded-challenge.zip?download=true`
    *   Extract the contents. You should have a folder containing the videos. **Ensure this folder is named `videos_full`** and place it in the root project directory. The metadata script expects these videos to exist locally.

7.  **Hugging Face Login:**
    *   You need to be logged into Hugging Face to download the dataset in Step 1 of the reproduction process. Run the following in your terminal:
        ```bash
        huggingface-cli login
        ```
    *   Follow the prompts to enter your Hugging Face token.

## How to Reproduce Results

Execute the following steps sequentially from the root directory of the cloned repository. Ensure your virtual environment is activated, the `.env` file is correctly configured, you are logged into Hugging Face (`huggingface-cli login`), and the `videos_full/` directory exists with the pre-downloaded videos.

**Step 1: Fetch Initial Metadata & Create Challenge Data CSV**

*   **Command:**
    ```bash
    python download_vids_metadata.py
    ```
*   **Action:**
    *   Downloads the `lmms-lab/AISG_Challenge` dataset from Hugging Face.
    *   Saves the test split into `challenge_data.csv`.
    *   Iterates through the unique YouTube URLs found in `challenge_data.csv`.
    *   For each URL, retrieves metadata (title, description, captions, etc.) using `pytubefix`. **Does not re-download videos.**
    *   Saves the collected metadata to `video_full_metadata.csv`.
    *   Logs any errors encountered during metadata fetching to `video_errors.csv`.
*   **Input:** Requires Hugging Face login, expects videos in `videos_full/`.
*   **Output:** `challenge_data.csv`, `video_full_metadata.csv`, `video_errors.csv`.

**Step 2: Scrape YouTube Comments**

*   **Command:**
    ```bash
    python yt_comment_scraper.py --input video_full_metadata.csv --output comments --comments 50 --verbose
    ```
*   **Input:** `video_full_metadata.csv` (generated in Step 1).
*   **Action:** Uses the YouTube Data API v3 to fetch the top 50 relevant comments for each video listed in the metadata file. Requires `YOUTUBE_API_KEY` in `.env`.
*   **Output:** Creates a `comments/` directory containing individual `.csv` files (named `{video_id}.csv`) for each video's comments.

**Step 3: Create Slowed-Down Videos**

*   **Command:**
    ```bash
    python create_slower_vids.py --input_metadata video_full_metadata.csv --input_dir videos_full --output_dir videos_full_slow
    ```
    *(Note: You might need to add argument parsing to `create_slower_vids.py` if it doesn't already support these flags. Assuming default paths match if flags aren't implemented: `python create_slower_vids.py`)*
*   **Input:**
    *   `video_full_metadata.csv` (for paths and captions).
    *   Videos located in `videos_full/`.
*   **Action:** Uses `ffmpeg` to create a 0.5x speed version of each video found in `videos_full/`. Adjusts SRT caption timestamps from the input metadata accordingly.
*   **Output:**
    *   Creates a `videos_full_slow/` directory containing the slowed-down `.mp4` files.
    *   Creates `videos_full_slow_metadata.csv` which is a copy of the input metadata but with `video_path` updated to point to the slowed videos and `caption` updated with adjusted timestamps.

**Step 4: Run Gemini Inference**

*   **Command:**
    ```bash
    python gemini.py \
      --challenge_data_path challenge_data.csv \
      --video_metadata_path videos_full_slow_metadata.csv \
      --comments_dir comments \
      --output_csv_path results/challenge_data_gemini_pro_2.5_exp_slowVid_v3_full.csv \
      --model_name models/gemini-2.5-pro-exp-03-25 \
      --v3 \
      --temperature 1.0 \
      # Add --rate_limit 2 if needed, e.g., 2 seconds between API calls
      # Remove --debug flag if present to run on all data
    ```
*   **Input:**
    *   `challenge_data.csv` (containing questions and linking `youtube_url` to `qid`).
    *   `videos_full_slow_metadata.csv` (from Step 3).
    *   `comments/` directory (from Step 2).
    *   Slowed videos in `videos_full_slow/` (referenced via metadata).
*   **Action:** Iterates through `challenge_data.csv`. For each question (`qid`), it finds the corresponding slowed video and metadata, reads comments, constructs the `v3` prompt, uploads the video to Gemini, runs inference using `gemini-2.5-pro-exp-03-25`, parses the response, and saves results incrementally. Requires `GEMINI_API_KEY` in `.env`. Creates the `results/` directory if it doesn't exist.
*   **Output:** Creates `results/challenge_data_gemini_pro_2.5_exp_slowVid_v3_full.csv` containing the questions, original data, and the model's generated answers (including thinking steps, descriptions, and final answer).

**Step 5: Convert to Submission Format**

*   **Command:**
    ```bash
    python convert_to_upload_format.py \
      --input results/challenge_data_gemini_pro_2.5_exp_slowVid_v3_full.csv \
      --output submissions/pred_gemini_pro_2.5_exp_slowVid_v3_full.csv
    ```
*   **Input:** The raw results CSV from Step 4.
*   **Action:** Selects and formats the required columns (likely `qid` and the final `answer`) into the specific format required for uploading to the leaderboard platform. Creates the `submissions/` directory if it doesn't exist.
*   **Output:** Creates `submissions/pred_gemini_pro_2.5_exp_slowVid_v3_full.csv`. **This is the file you should upload to the challenge platform.**

## Benchmark Results

The final prediction results, formatted for submission, can be found in the `submissions/` directory, specifically in the file generated by Step 5 above (`submissions/pred_gemini_pro_2.5_exp_slowVid_v3_full.csv`).

*(Please refer to the challenge platform leaderboard for our official score.)*


