import os
import argparse
import pandas as pd
import google.generativeai as genai
import time
import csv
import re
from dotenv import load_dotenv
import prompts


def sanitize_filename(title):
    """
    Convert a title to a valid filename by removing invalid characters.
    """
    clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
    clean_title = clean_title.replace(" ", "_")[:100]
    return clean_title


def read_comments_file(filename, comments_dir):
    """
    Read comments from a CSV file in the comments directory.

    Parameters:
    -----------
    filename : str
        The filename of the CSV file containing comments
    comments_dir : str
        Directory where comment CSV files are stored

    Returns:
    --------
    list
        List of comment dictionaries
    """
    comments_path = os.path.join(comments_dir, filename)
    if not os.path.exists(comments_path):
        return []

    try:
        comments_df = pd.read_csv(comments_path)
        comments = []
        for _, row in comments_df.iterrows():
            comment_dict = {}
            if "Author" in row:
                comment_dict["Author"] = row["Author"]
            if "Comment" in row:
                comment_dict["Comment"] = row["Comment"]
            comments.append(comment_dict)
        return comments
    except Exception as e:
        print(f"Error reading comments file {comments_path}: {str(e)}")
        return []


def upload_and_process_video(video_path):
    """
    Upload a video to Gemini API and wait for processing to complete.

    Parameters:
    -----------
    video_path : str
        Path to the video file

    Returns:
    --------
    object or None
        Uploaded file object from Gemini API, or None if failed
    """
    try:
        print(f"Uploading video: {video_path}")
        video_file = genai.upload_file(path=video_path)

        print("Waiting for video to be processed...")
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")

        print("Video processed successfully.")
        return video_file
    except Exception as e:
        print(f"Error uploading/processing video: {str(e)}")
        return None


def main():
    """
    Main function to run the video inference pipeline.
    Uses command-line arguments to set file paths and debug mode.
    """

    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Run the Gemini video inference pipeline with optional overrides for file paths."
    )
    parser.add_argument(
        "--challenge_data_path",
        default="challenge_data.csv",
        help="Path to the challenge data CSV file (default: challenge_data.csv).",
    )
    parser.add_argument(
        "--video_metadata_path",
        default="video_metadata.csv",
        help="Path to the video metadata CSV file (default: video_metadata.csv).",
    )
    parser.add_argument(
        "--output_csv_path",
        default="results/challenge_data_gemini_debug.csv",
        help="Path to the output CSV file for storing inference results (default: challenge_data_gemini.csv).",
    )
    parser.add_argument(
        "--comments_dir",
        default="comments",
        help="Directory where comment CSV files are stored (default: comments).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, only the first few videos will be processed.",
    )
    # flags for prompt version
    parser.add_argument("--v2", action="store_true", help="If set, use the v2 prompt.")
    parser.add_argument("--v3", action="store_true", help="If set, use the v3 prompt.")
    # Temperature
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature value for sampling text generation (default: 0.5).",
    )
    # multi-vid: Adds multiple vids into prompt
    parser.add_argument(
        "--multi-vid",
        action="store_true",
        help="If set, use multiple videos in the prompt.",
    )
    # add caution_prompt
    parser.add_argument(
        "--caution_prompt",
        action="store_true",
        help="If set, add caution prompt to the prompt.",
    )
    # rate limit for gemini
    parser.add_argument(
        "--rate_limit",
        type=int,
        default=2,
        help="Rate limit for Gemini API (default: 0).",
    )
    # model name
    parser.add_argument(
        "--model_name",
        default="models/gemini-2.0-flash",
        help="Model name for Gemini API (default: models/gemini-2.0-flash).",
    )
    # detailed response
    parser.add_argument(
        "--detailed_response",
        action="store_true",
        help="If set, induce detailed response.",
    )
    # no prompt
    parser.add_argument(
        "--no_prompt", action="store_true", help="If set, do not use prompt."
    )
    parser.add_argument(
        "--filter_video_id",
        type=str,
        help="Process only videos containing this ID",
    )
    args = parser.parse_args()

    # print args in a nice format
    # print(f"Arguments:")
    # for arg in vars(args):
    #     print(f"  {arg}: {getattr(args, arg)}")

    # -----------------------------
    # Load environment and configure Gemini
    # -----------------------------
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    genai.configure(api_key=api_key)

    # -----------------------------
    # Ensure directories exist
    # -----------------------------
    if not os.path.exists(args.comments_dir):
        os.makedirs(args.comments_dir)

    # -----------------------------
    # Read input CSV files
    # -----------------------------
    # print temperature and prompt version
    print(f"Temperature: {args.temperature}")
    if args.v2:
        print("Using v2 prompt.")
    elif args.v3:
        print("Using v3 prompt.")
    else:
        print("Using default prompt.")

    print(f"Reading challenge data from {args.challenge_data_path}")
    df_challenge_data = pd.read_csv(args.challenge_data_path).copy()

    print(f"Reading video metadata from {args.video_metadata_path}")
    df_video_metadata = pd.read_csv(args.video_metadata_path).copy()

    # Then in the processing code:
    if args.filter_video_id:
        df_challenge_data = df_challenge_data[df_challenge_data["youtube_url"].str.contains(args.filter_video_id)]

    # Add all needed answer columns if missing
    if "answer" not in df_challenge_data.columns:
        df_challenge_data["answer"] = ""
    if "thinking_steps" not in df_challenge_data.columns:
        df_challenge_data["thinking_steps"] = ""
    if "video_start" not in df_challenge_data.columns:
        df_challenge_data["video_start"] = ""
    if "video_middle" not in df_challenge_data.columns:
        df_challenge_data["video_middle"] = ""
    if "video_end" not in df_challenge_data.columns:
        df_challenge_data["video_end"] = ""

    # -----------------------------
    # Resume from existing output CSV if present
    # -----------------------------
    if os.path.exists(args.output_csv_path):  # <-- Added for resumption
        print(
            f"Found existing output CSV at {args.output_csv_path}. Resuming from last saved row."
        )
        df_existing = pd.read_csv(args.output_csv_path)
        # For any columns that overlap, copy over existing data
        for col in [
            "answer",
            "thinking_steps",
            "video_start",
            "video_middle",
            "video_end",
        ]:
            if col in df_existing.columns:
                df_challenge_data[col] = df_existing[col]

    processed_videos = {}

    # -----------------------------
    # Set up the Gemini model
    # -----------------------------
    model = genai.GenerativeModel(
        model_name=args.model_name, generation_config={"temperature": args.temperature}
    )

    # -----------------------------
    # Process each entry
    # -----------------------------
    max_videos = 5  # how many videos to process in debug mode

    for idx, row in df_challenge_data.iterrows():
        # Skip if there's already an answer from a previous run
        if pd.notna(row["answer"]) and row["answer"].strip() != "":
            print(f"Skipping QID={row['qid']} as it already has an answer.")
            continue

        # If debug mode is on and we've processed enough examples, stop
        if args.debug and idx >= max_videos:
            print("\n[DEBUG] Reached debug limit of videos. Stopping early.")
            break

        # try:
        youtube_url = row["youtube_url"]
        qid = row["qid"]

        print(f"\nProcessing QID={qid} with youtube_url={youtube_url}")

        # Combine question and question_prompt
        question = row.get("question", "")
        question_prompt = row.get("question_prompt", "")
        combined_question = f"{question} {question_prompt}".strip()

        # Find matching metadata
        matching_metadata = df_video_metadata[
            df_video_metadata["youtube_url"] == youtube_url
        ]
        if matching_metadata.empty:
            print(f"No metadata found for youtube_url: {youtube_url}")
            continue

        metadata = matching_metadata.iloc[0]
        video_path = metadata["video_path"]

        if not os.path.exists(video_path):
            print(f"Video file not found at path: {video_path}")
            continue

        video_title = metadata.get("title", "")
        video_description = metadata.get("description", "")
        captions = metadata.get("caption", "")
        channel_name = metadata.get("channel_name", "")
        video_id = metadata.get("video_id", "")

        # Read comments file
        # clean_title = sanitize_filename(video_title)
        clean_title = video_title
        comments_filename = f"{video_id}.csv"
        comments = read_comments_file(comments_filename, args.comments_dir)
        # print check to see if comments are retrieved successfully

        print(f"Retrieved comments for video ID={video_id}: {comments}")

        # Choose the appropriate prompt version
        if args.v3:
            prompt_text = prompts.create_video_qa_prompt_v3(
                video_title=video_title,
                video_description=video_description,
                comments=comments,
                captions=captions if captions else None,
                channel_name=channel_name,
                question=combined_question,
                caution_prompt=args.caution_prompt,
                detailed_response=args.detailed_response,
            )
        elif args.v2:
            prompt_text = prompts.create_video_qa_prompt_v2(
                video_title=video_title,
                video_description=video_description,
                comments=comments,
                captions=captions if captions else None,
                channel_name=channel_name,
                question=combined_question,
            )
        else:
            prompt_text = prompts.create_video_qa_prompt(
                video_title=video_title,
                video_description=video_description,
                comments=comments,
                captions=captions if captions else None,
                channel_name=channel_name,
                question=combined_question,
            )
        if args.no_prompt:
            prompt_text = combined_question

        if args.debug:
            # print url and question text
            print(f"URL: {youtube_url}")
            print(f"Question text for QID={qid}:")
            # print prompt text
            print(f"Prompt text for QID={qid}:")
            print("-" * 50)
            print(prompt_text)
            print("-" * 50)

        # Upload and process the video if not already done
        if video_path in processed_videos:
            video_file = processed_videos[video_path]
            print(f"Using previously uploaded video for {video_path}")
        else:
            video_file = upload_and_process_video(video_path)
            if video_file:
                processed_videos[video_path] = video_file
            else:
                print(f"Skipping video due to processing error: {video_path}")
                continue

        # Generate response from Gemini
        print("Making Gemini inference request...")
        input_data = [prompt_text, video_file]
        if args.multi_vid:
            input_data = [video_file, prompt_text, video_file]
        response = model.generate_content(
            # tried [video_file, prompt_text] but resulted in perf. drop
            # from 47.53% --> 46.13% in v3-prompt + slow-vid seting
            input_data,
            request_options={"timeout": 600},
        )

        # sleep for rate limit
        if args.rate_limit > 0:
            print(f"Sleeping for {args.rate_limit} seconds...")
            time.sleep(args.rate_limit)

        try:
            answer = response.text.strip()
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            answer = str(response)
            print(f"Response: {answer}")
            continue

        # Initialize all fields
        video_start = ""
        video_middle = ""
        video_end = ""
        thinking_steps = ""
        final_answer = ""

        # Extract fields based on prompt version
        if args.v3:
            # Try to extract all five fields from v3 output
            start_match = re.search(
                r"VIDEO START DESCRIPTION:(.*?)(?=VIDEO MIDDLE DESCRIPTION:|$)",
                answer,
                re.DOTALL,
            )
            if start_match:
                video_start = start_match.group(1).strip()

            middle_match = re.search(
                r"VIDEO MIDDLE DESCRIPTION:(.*?)(?=VIDEO END DESCRIPTION:|$)",
                answer,
                re.DOTALL,
            )
            if middle_match:
                video_middle = middle_match.group(1).strip()

            end_match = re.search(
                r"VIDEO END DESCRIPTION:(.*?)(?=THINKING STEPS:|$)", answer, re.DOTALL
            )
            if end_match:
                video_end = end_match.group(1).strip()

            thinking_match = re.search(
                r"THINKING STEPS:(.*?)(?=FINAL ANSWER:|$)", answer, re.DOTALL
            )
            if thinking_match:
                thinking_steps = thinking_match.group(1).strip()

            answer_match = re.search(r"FINAL ANSWER:(.*?)$", answer, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                print(
                    "Warning: Could not extract 'FINAL ANSWER:' from the text. Falling back to entire answer."
                )
                final_answer = answer

        elif args.v2:
            # Extract just thinking and answer from v2 output
            thinking_match = re.search(
                r"THINKING STEPS:(.*?)(?=FINAL ANSWER:|$)", answer, re.DOTALL
            )
            if thinking_match:
                thinking_steps = thinking_match.group(1).strip()

            answer_match = re.search(r"FINAL ANSWER:(.*?)$", answer, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                print(
                    "Warning: Could not extract 'FINAL ANSWER:' from the text. Falling back to entire answer."
                )
                final_answer = answer
        else:
            # For v1, just use the entire answer
            final_answer = answer

        # Store all extracted fields
        df_challenge_data.at[idx, "video_start"] = video_start
        df_challenge_data.at[idx, "video_middle"] = video_middle
        df_challenge_data.at[idx, "video_end"] = video_end
        df_challenge_data.at[idx, "thinking_steps"] = thinking_steps
        df_challenge_data.at[idx, "answer"] = final_answer

        print(f"Generated answer for QID={qid}:")
        print("-" * 50)
        # print question
        print(f"Question: {combined_question}")
        # if v3, print all fields
        if args.v3:
            print("-" * 50)
            print(f"VIDEO START DESCRIPTION: {video_start}")
            print("-" * 50)
            print(f"VIDEO MIDDLE DESCRIPTION: {video_middle}")
            print("-" * 50)
            print(f"VIDEO END DESCRIPTION: {video_end}")
            print("-" * 50)
            print(f"THINKING STEPS: {thinking_steps}")
        # v2, print thinking and answer
        elif args.v2:
            print("-" * 50)
            print(f"THINKING STEPS: {thinking_steps}")

        print("-" * 50)
        print(final_answer[:200] + "..." if len(final_answer) > 200 else final_answer)
        print("-" * 50)

        # Save partial progress after each row  # <-- Added for resumption
        df_challenge_data.to_csv(args.output_csv_path, index=False)

    # -----------------------------
    # Final save (though we've been saving after each iteration)
    # -----------------------------
    print(f"\nSaving results to {args.output_csv_path}")
    df_challenge_data.to_csv(args.output_csv_path, index=False)
    print(f"Inference pipeline completed. Answers saved to: {args.output_csv_path}")


if __name__ == "__main__":
    main()
