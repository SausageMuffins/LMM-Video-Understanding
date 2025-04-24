import os
import json
import sys
import time
import pandas as pd
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from vllm import LLM, SamplingParams  # vLLM is used for GPU-accelerated inference
import numpy as np
import google.generativeai as genai  # Gemini API library for video understanding

# Set environment variable to help avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Monkey-patch for NumPy 2.0 compatibility:
if not hasattr(np.ndarray, "newbyteorder"):
    def newbyteorder(self, order=None):
        """
        A monkey-patch for np.ndarray.newbyteorder.
        Returns a view with the data-type bytes swapped using the given order.
        """
        return self.view(self.dtype.newbyteorder(order))
    np.ndarray.newbyteorder = newbyteorder

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API using google.generativeai
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

def download_model():
    """
    Download the DeepSeek-R1-Distill-Qwen-32B-Q6_K.gguf model from Hugging Face.
    Returns the local path to the model.
    """
    repo_id = "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"
    filename = "DeepSeek-R1-Distill-Qwen-32B-Q6_K.gguf"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Model downloaded to: {model_path}")
    return model_path

def upload_and_process_video(video_path: str):
    """
    Upload a video file to Gemini API and poll until processing is complete.
    
    Parameters:
        video_path (str): The local path to the video file.
    
    Returns:
        object: The processed video file object from Gemini API.
    """
    try:
        print(f"Uploading video: {video_path}")
        video_file = genai.upload_file(path=video_path)  # Upload the video file
        
        print("Waiting for video processing...")
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)  # Poll for updated status

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed for {video_path}: {video_file.state.name}")

        print("Video processed successfully.")
        return video_file
    except Exception as e:
        print(f"Error uploading/processing video {video_path}: {str(e)}", file=sys.stderr)
        return None

def generate_overview(video_path: str, qn: str) -> str:
    """
    Generate an overview of the video content using Gemini 2.0 Flash.
    The overview is a one-paragraph summary capturing the key points.
    
    Parameters:
        video_path (str): Path to the video file.
        qn (str): The guiding text prompt for generating a more informed overview.
    
    Returns:
        str: The generated overview text.
    """
    # Upload and process the video via Gemini
    video_file = upload_and_process_video(video_path)
    if video_file is None:
        return ""
    
    # Build the prompt for generating the overview.
    prompt = (
        f"Provide a detailed one-paragraph overview of the video content, highlighting the main actions and context. In your response, also include the counts of the entities/events and attributes like the perspective/viewpoints of entities/events involved in the video. You may consider the following prompt to help guide your response:\n {qn}"
    )
    input_data = [prompt, video_file]
    
    # Initialize the Gemini 2.0 Flash model.
    model = genai.GenerativeModel(
        model_name="models/gemini-2.0-flash",
        generation_config={"temperature": 0.6}  # Deterministic output for overview
    )
    # Request the content with an extended timeout.
    response = model.generate_content(input_data, request_options={"timeout": 600})
    try:
        # Retrieve the overview from the top-level text field.
        overview_text = response.text.strip()
        print("Generated overview:")
        print(overview_text)
    except Exception as e:
        print(f"Error generating overview: {e}", file=sys.stderr)
        overview_text = ""
    return overview_text

def generate_descriptions(prompt: str, overview: str, llm: LLM) -> dict:
    """
    Generate detailed video scene descriptions based on the original text prompt and the generated overview.
    
    Returns:
        dict: A JSON object containing:
          "expected_scenes", "unexpected_scenes", and "generalized_scenes".
    """
    instruction = f"""Given the following video understanding prompt:
    "{prompt}"

    and the following one-paragraph overview of the video content:
    "{overview}"

    Generate a JSON object that contains three keys:
    "expected_scenes": a list of expected scenes related to the video understanding prompt.
    "unexpected_scenes": a list of unexpected scenes (the opposite of expected scenes).
    "generalized_scenes": a list of generalized scenes in the video.

    For example, given the video understanding prompt: What is the difference between the action of the last person in the video and the actions of the first two people? 
            
    and the narrative overview: The video shows three people in a park. The first two people that appear in the video are playing frisbee, while the last person to appear in the video is sitting on a bench and reading a book. The first two people are having fun, while the last person is focused on their book. The first two people are active, while the last person is passive. The first two people are socializing, while the last person is alone. The first two people are enjoying the outdoors, while the last person is indoors.
            
    The expected scenes could be:
    "expected_scenes": [
        "The first two people are playing frisbee in a park.",
        "The last person is sitting on a bench reading a book.",
        "The actions of the first two and the last are the same"
    ],

    The unexpected scenes could be:
    "unexpected_scenes": [
        "The first two people are not playing frisbee in a park.",
        "The last person is not sitting on a bench reading a book.",
        "The actions of the first two and the last are not the same"
    ],

    The generalized scenes could be:
    "generalized_scenes": [
        "The first two people are enjoying the outdoors.",
        "The last person is enjoying the indoors."
    ]

    Another example might involve the counting the number of entities in the video such as the number of people doing the activity.

    Write your final answer as a json object without the use of backticks (important):"""+"""\n
    Final Answer:
    {
        "expected_scenes": [
            "scene 1"
        ],
        "unexpected_scenes": [
            "scene 1"
        ],
        "generalized_scenes": [
            "scene 1"
        ]
    }
    """
    # Define sampling parameters for vLLM.
    params = SamplingParams(
        max_tokens=4096,
        temperature=0.6,
        repetition_penalty=1.0
    )
    # Generate output using the vLLM model with GPU acceleration.
    output=""
    output = llm.generate(instruction, sampling_params=params)
    # Retrieve the text from the first CompletionOutput in the first RequestOutput.
    generated_text = output[0].outputs[0].text
    print("Raw model output:")
    print(generated_text)

    # Robust JSON extraction:
    json_start = generated_text.find("{")
    if json_start == -1:
        raise ValueError("No JSON object found in the generated output.")
    json_end = generated_text.rfind("}")
    if json_end == -1:
        raise ValueError("No closing bracket found in the generated output.")
    json_str = generated_text[json_start:json_end+1]
    try:
        data = json.loads(json_str)
        return data
    except Exception as e:
        raise ValueError(f"Error parsing generated output: {e}\nOutput was: {generated_text}")

def rerun_failed_entries(all_outputs: dict, df: pd.DataFrame, llm: LLM, overview_cache: dict, error_log_file: str) -> dict:
    """
    Rerun the generation of JSON descriptions for entries in all_outputs that contain errors.
    
    For each entry with an error, re-read the corresponding row from df,
    re-calculate (or retrieve cached) overview using the question from the row with qid ending in '-0'
    (for the same video) as guiding text, and then re-run generate_descriptions.
    If the rerun is successful, directly update the entry in all_outputs (thus removing the error).
    Any exceptions are logged to error_log_file.
    
    Returns:
        dict: The updated all_outputs dictionary with retried entries.
    """
    # Create a lookup mapping from qid to its row.
    qid_to_row = df.set_index("qid").to_dict(orient="index")
    for index, row in df.iterrows():
        qid = row["qid"]
        if qid not in all_outputs or "error" not in all_outputs[qid]:
            continue  # Skip entries that do not have errors
        
        prompt_text = row["question"]
        raw_video_id = row["video_id"]
        video_id = raw_video_id.strip().strip('"')
        video_path = f"data/{video_id}.mp4"
        
        # Use the guiding question from the row with the same prefix and qid ending in '-0'
        prefix = qid.split('-')[0]
        guiding_qid = f"{prefix}-0"
        if guiding_qid in qid_to_row:
            guiding_prompt = qid_to_row[guiding_qid]["question"]
        else:
            guiding_prompt = f"{video_id}-0"
        
        print(f"\nRe-running failed entry for QID: {qid} | Video: {video_path} | Using guiding question from QID {guiding_qid}:")
        print(guiding_prompt)
        
        try:
            if video_id in overview_cache:
                overview = overview_cache[video_id]
                print(f"Using cached overview for video ID {video_id}:")
                print(overview)
            else:
                overview = generate_overview(video_path, qn=guiding_prompt)
                overview_cache[video_id] = overview
            
            scenes = generate_descriptions(prompt_text, overview, llm)
            combined_output = {
                "overview": overview,
                "expected_scenes": scenes.get("expected_scenes", []),
                "unexpected_scenes": scenes.get("unexpected_scenes", []),
                "generalized_scenes": scenes.get("generalized_scenes", [])
            }
            # Update all_outputs directly, replacing the error.
            all_outputs[qid] = combined_output
            print(f"Successfully re-ran entry for QID {qid}.")
        except Exception as e:
            error_message = f"Rerun failed for QID {qid} (Video ID {video_id}): {e}"
            print(error_message, file=sys.stderr)
            with open(error_log_file, "a") as elog:
                elog.write(error_message + "\n")
    return all_outputs

def main():
    """
    Main entry point for re-running error entries.
    
    Loads the CSV file 'challenge_data_corrected.csv', which contains columns for question, qid, 
    and video_id (wrapped with extra quotes), and also loads the initial JSON output file 
    'json_descriptions.json'. For each entry with errors:
      - Removes extra quotation marks from the video_id.
      - Constructs the video file path as 'data/{video_id}.mp4'.
      - Uses the question from the row with qid ending in '-0' as guiding text for generating (or re-using a cached) overview via Gemini API.
      - Re-runs detailed scene description generation via vLLM using the question prompt and the overview.
    Directly updates the original JSON file so that entries with errors are replaced by the new descriptions.
    Additionally, saves a separate "error.json" file for all entries that still have errors. Each error
    entry includes the generated overview and the error message with the deepseek model output.
    """
    csv_path = "challenge_data_corrected.csv"
    output_path = "json_descriptions.json"
    error_output_path = "error.json"
    
    # Load the CSV file.
    df = pd.read_csv(csv_path)
    
    # Load the initial outputs from the JSON file.
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_outputs = json.load(f)
    else:
        print("Initial output file not found. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Initialize the vLLM model with reduced maximum sequence length and eager mode enabled.
    llm_model_path = "./DeepSeek-R1-Distill-Qwen-32B-Q6_K.gguf"
    llm = LLM(model=llm_model_path, device="cuda", max_model_len=1500, enforce_eager=True)
    
    # Initialize a dictionary to cache generated overviews (keyed by video_id)
    overview_cache = {}
    
    # Initialize an error log file.
    error_log_file = "failed_videos.txt"
    with open(error_log_file, "w") as elog:
        elog.write("")
    
    # Only re-run entries that contain an error.
    all_outputs = rerun_failed_entries(all_outputs, df, llm, overview_cache, error_log_file)
    
    # Update the original JSON file with the new outputs.
    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=2)
    print(f"\nOriginal JSON file updated: {output_path}")
    
    # Build error entries dictionary: include entries that still have errors along with their overview.
    error_entries = {}
    qid_to_row = df.set_index("qid").to_dict(orient="index")
    for qid, entry in all_outputs.items():
        if "error" in entry:
            # If no overview is present in the error entry, try to obtain it from the cache using the video_id.
            if "overview" not in entry:
                if qid in qid_to_row:
                    raw_video_id = qid_to_row[qid]["video_id"]
                    video_id = raw_video_id.strip().strip('"')
                    if video_id in overview_cache:
                        entry["overview"] = overview_cache[video_id]
            error_entries[qid] = entry
    
    # Save error entries to a separate JSON file.
    with open(error_output_path, "w") as ef:
        json.dump(error_entries, ef, indent=2)
    print(f"\nError JSON file saved: {error_output_path}")

if __name__ == "__main__":
    main()
