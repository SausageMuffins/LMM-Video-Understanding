import os
import sys
import json
import pandas as pd

from vllm import LLM, SamplingParams

def generate_answer(prompt: str, overview: str, llm) -> str:
    """
    Generate a detailed answer using the DeepSeek R1 model.
    
    This function creates an instruction by combining the prompt, the video overview,
    and an instruction to structure the final answer with "Final Answer:".
    
    Args:
        prompt (str): The question prompt constructed from the CSV.
        overview (str): The overview description of the video from the JSON file.
        llm (LLM): The loaded DeepSeek R1 model.
    
    Returns:
        str: The final generated answer text.
    """
    # Construct the instruction with the desired structure.
    instruction = (
        f"Use the overview of the video content to answer the following prompt:\n"
        f"\"{prompt}\"\n\n"
        f"Video Overview:\n"
        f"\"{overview}\"\n\n"
        f"Structure your final answer after the reasoning process as:\nFinal Answer: "
    )
    
    # Define sampling parameters for vLLM.
    params = SamplingParams(
        max_tokens=4096,     # maximum number of tokens to generate
        temperature=0.6,     # set the temperature for sampling
        repetition_penalty=1.0  # repetition penalty parameter
    )
    
    # Generate the output using the model
    output = llm.generate(instruction, sampling_params=params)
    generated_text = output[0].outputs[0].text  # retrieve generated answer text
    # Optionally, further processing can be done if needed to extract the answer.
    return generated_text

def main():
    """
    Main function to load challenge data, generate answers using DeepSeek R1, and save the submission.
    
    Steps:
      1. Load the 'challenge_data_corrected.csv' containing all questions.
      2. Load the JSON file 'json_descriptions.json' which holds video overviews keyed by qid.
      3. For each entry, combine the 'question' and 'question_prompt' fields into a prompt.
      4. Use deepseek r1 to generate an answer for the prompt, including an instruction to structure the answer.
      5. Save the generated answers into a new CSV file 'submission.csv' with columns for 'qid' and 'answer'.
    """
    c
    rrected.csv"
    json_path = "json_descriptions.json"
    submission_path = "submission.csv"
    
    # Load the CSV file with questions.
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load the JSON file with video overviews.
    try:
        with open(json_path, "r") as f:
            overviews = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize the DeepSeek R1 model.
    # The model file and parameters can be adjusted as necessary.
    llm_model_path = "./DeepSeek-R1-Distill-Qwen-32B-Q6_K.gguf"
    llm = LLM(model=llm_model_path, device="cuda", max_model_len=2048, enforce_eager=True)  # DeepSeek R1 model
    
    # Prepare a list to hold the results.
    results = []
    
    # Process each row in the CSV.
    for index, row in df.iterrows():
        qid = row["qid"]
        question = row["question"]
        question_prompt = row["question_prompt"]
        
        # Combine question and question_prompt into one prompt string.
        prompt = f"{question} {question_prompt}"
        
        # Retrieve the overview for the qid from the JSON file.
        # If no overview is present, use an empty string.
        overview = ""
        if qid in overviews and "overview" in overviews[qid]:
            overview = overviews[qid]["overview"]
        
        try:
            # Generate the answer using the DeepSeek R1 model.
            answer = generate_answer(prompt, overview, llm)
        except Exception as e:
            print(f"Error generating answer for qid {qid}: {e}", file=sys.stderr)
            answer = "Error generating answer."
        
        # Append the qid and its answer to the result list.
        results.append({"qid": qid, "answer": answer})
    
    # Save the results to submission.csv
    try:
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved successfully: {submission_path}")
    except Exception as e:
        print(f"Error saving submission file: {e}", file=sys.stderr)
    
if __name__ == "__main__":
    main()
