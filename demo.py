import streamlit as st  # UI library for building interactive web apps
import subprocess  # Run shell commands and external processes
import os  # Interact with the operating system
import pandas as pd  # For loading CSV comments
import re  # For regex pattern matching
import json  # For serializing questions to pass to the inference script


def run_step(command, step_name, video_id):
    """
    Execute a shell command and display its live output in the Streamlit app.

    Args:
        command (str): The shell command to execute.
        step_name (str): A descriptive name of the step for UI display.
        video_id (str): The YouTube video ID being processed.
    """
    st.write(f"### {step_name}")
    # Prepare a container to accumulate and display logs
    log_container = st.empty()
    logs = ""

    with st.spinner(f"Running {step_name}..."):
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # Stream and accumulate stdout
            for line in process.stdout:
                logs += line
                # Display accumulated output in a scrollable text area
                log_container.text_area("Output", value=logs, height=250)
            process.wait()
            # Final status
            if process.returncode == 0:
                # showing the video
                if step_name == "Step 0: Download metadata":
                    video_path = os.path.join("demo", "videos_full", f"{video_id}.mp4")
                    if os.path.exists(video_path):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.video(video_path)
                            st.caption(f"Original Video (ID: {video_id})")

                # show the top 10 comments
                st.success(f"{step_name} completed successfully.")
                if step_name == "Step 1: Scrape YouTube comments":
                    comments_file = os.path.join("demo", "comments", f"{video_id}.csv")
                    if os.path.exists(comments_file):
                        df_comments = pd.read_csv(comments_file)
                        st.write("Top 10 Comments:")
                        st.table(df_comments.head(10))

                # show the slowed video
                elif step_name == "Step 2: Create slowed-down videos":
                    video_path = os.path.join(
                        "demo", "videos_full_slow", f"{video_id}.mp4"
                    )
                    if os.path.exists(video_path):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.video(video_path)
                            st.caption(f"Slowed Video (ID: {video_id})")
            else:
                st.error(f"{step_name} failed with exit code {process.returncode}.")
        except Exception as e:
            st.error(f"Error running {step_name}: {e}")


def display_formatted_output(video_id):
    """
    Display the formatted output from the Gemini inference results.
    
    Args:
        video_id (str): The YouTube video ID being processed.
    """
    results_path = os.path.join("demo", "results", "demo_gemini_output.csv")
    if not os.path.exists(results_path):
        st.warning("No inference results available yet. Run Step 3 first.")
        return
    
    try:
        df = pd.read_csv(results_path)
        # Filter for the specific video ID
        # First find the YouTube URL that contains this video ID
        url_pattern = f".*{video_id}.*"
        matching_rows = df[df['youtube_url'].str.contains(url_pattern, regex=True, na=False)]
        
        if matching_rows.empty:
            st.warning(f"No results found for video ID: {video_id}")
            return
            
        # Display each row's results
        for idx, row in matching_rows.iterrows():
            st.subheader(f"Question: {row.get('question', '')} {row.get('question_prompt', '')}")
            
            # Create expandable sections for each part of the output
            with st.expander("Video Start Description", expanded=True):
                st.write(row.get('video_start', 'Not available'))
                
            with st.expander("Video Middle Description", expanded=True):
                st.write(row.get('video_middle', 'Not available'))
                
            with st.expander("Video End Description", expanded=True):
                st.write(row.get('video_end', 'Not available'))
                
            with st.expander("Thinking Steps", expanded=True):
                st.write(row.get('thinking_steps', 'Not available'))
                
            with st.expander("Final Answer", expanded=True):
                st.write(row.get('answer', 'Not available'))
    
    except Exception as e:
        st.error(f"Error displaying results: {e}")


def extract_questions(video_id):
    """
    Extract questions from challenge data for a specific video ID.
    
    Args:
        video_id (str): The YouTube video ID to find questions for.
    
    Returns:
        list: List of question dictionaries associated with the video ID
    """
    challenge_path = os.path.join("demo", "challenge_data.csv")
    if not os.path.exists(challenge_path):
        return []
    
    try:
        df = pd.read_csv(challenge_path)
        # Filter for the specific video ID
        url_pattern = f".*{video_id}.*"
        matching_rows = df[df['youtube_url'].str.contains(url_pattern, regex=True, na=False)]
        
        questions = []
        for _, row in matching_rows.iterrows():
            question_text = f"{row.get('question', '')} {row.get('question_prompt', '')}"
            question_dict = {
                'qid': row.get('qid', 'unknown'),
                'text': question_text.strip(),
                'type': row.get('question_type', 'Custom'),
                'source': 'challenge_data',
                'youtube_url': row.get('youtube_url', '')
            }
            questions.append(question_dict)
        
        return questions
    except Exception as e:
        st.error(f"Error extracting questions: {e}")
        return []


def get_video_url_from_id(video_id):
    """Generate a YouTube URL from a video ID."""
    # Handle both full URLs and just IDs
    if "youtube.com" in video_id or "youtu.be" in video_id:
        return video_id  # It's already a URL
    
    # Check if it looks like a YouTube short
    if len(video_id) == 11:  # Standard YouTube video ID length
        return f"https://www.youtube.com/shorts/{video_id}"
    
    return video_id  # Return as-is if we can't determine format


def create_temp_challenge_data(questions, video_id, output_path="demo/temp_challenge_data.csv"):
    """
    Create a temporary challenge data CSV file with only the selected questions.
    
    Args:
        questions (list): List of selected question dictionaries.
        video_id (str): The YouTube video ID.
        output_path (str): Path to save the temporary CSV file.
        
    Returns:
        str: Path to the temporary CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a dataframe with the necessary columns
    df = pd.DataFrame(columns=['qid', 'question', 'question_prompt', 'youtube_url'])
    
    for i, q in enumerate(questions):
        # For custom questions, create a temporary qid
        if q.get('source') == 'custom':
            qid = f"custom-{i}"
        else:
            qid = q.get('qid', f"unknown-{i}")
        
        # Split the question text into question and prompt if possible
        # Default approach: treat entire text as question
        question_text = q.get('text', '')
        question_prompt = ""
        
        # If there's a "Please state..." or similar prompt at the end, separate it
        prompt_match = re.search(r'(Please state.+)$', question_text)
        if prompt_match:
            question_prompt = prompt_match.group(1)
            question_text = question_text.replace(question_prompt, '').strip()
        
        # Add row to dataframe
        df.loc[i] = {
            'qid': qid,
            'question': question_text,
            'question_prompt': question_prompt,
            'youtube_url': q.get('youtube_url', get_video_url_from_id(video_id))
        }
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    return output_path


def main():
    """
    Define the Streamlit UI layout and handle user interactions to trigger pipeline steps.
    """
    st.title("AISG NAISC TikTok Challenge")
    st.write(
        "Enter a YouTube video ID and use the buttons to run individual steps or execute the entire pipeline."
    )

    # Video ID input
    if 'video_id' not in st.session_state:
        st.session_state.video_id = "HsXS1Qt11cU"  # Default value
    
    # Initialize questions in session state if not present
    if 'questions' not in st.session_state:
        st.session_state.questions = []
        st.session_state.selected_questions = []
    
    video_id = st.text_input(
        "Enter YouTube Video ID:",
        value=st.session_state.video_id,
        help="Enter an 11-character YouTube video ID (e.g., HsXS1Qt11cU)",
    )
    
    # Save to session state when changed
    if video_id != st.session_state.video_id:
        st.session_state.video_id = video_id
        # Reset questions when video ID changes
        st.session_state.questions = extract_questions(video_id)
        st.session_state.selected_questions = []
    
    st.sidebar.subheader("Debug Info")
    st.sidebar.text(f"Current Video ID: {video_id}")
    
    video_url = get_video_url_from_id(video_id)
    
    # Display the URL and a button to open it
    st.write(f"YouTube URL: {video_url}")
    st.markdown(f"[Open in YouTube]({video_url})", unsafe_allow_html=True)
    
    # Map descriptive step names to their shell commands with dynamic video ID
    steps = {
        "Step 0: Download metadata": (
            f"mkdir -p demo && "
            f"(cd demo && python ../demo_download_vids_metadata.py {video_id}) && "
            f"mkdir -p demo/comments demo/videos_full_slow demo/results demo/submissions"
        ),
        "Step 1: Scrape YouTube comments": (
            f"python yt_comment_scraper.py "
            f"--input demo/video_full_metadata.csv "
            f"--output demo/comments "
            f"--comments 50 "
            f"--verbose"
        ),
        "Step 2: Create slowed-down videos": (
            f"python demo_create_slower_vids.py "
            f"--input_metadata demo/video_full_metadata.csv "
            f"--input_dir demo/videos_full "
            f"--output_dir demo/videos_full_slow "
            f"--video_id {video_id}"
        ),
        "Step 3: Run Gemini inference": (
            f"python gemini.py "
            f"--challenge_data_path {{challenge_data_path}} "  # Placeholder to be filled later
            f"--video_metadata_path demo/video_full_metadata.csv "
            f"--comments_dir demo/comments "
            f"--output_csv_path demo/results/demo_gemini_output.csv "
            f"--model_name models/gemini-2.5-pro-preview-05-06 "
            f"--v3 "
            f"--temperature 1.0 "
            f"--filter_video_id {video_id}"
        ),
        "Step 4: Submission Format": (
            f"python convert_to_upload_format.py "
            f"--input demo/results/demo_gemini_output.csv "
            f"--output demo/submissions/demo_submission.csv"
        ),
    }
    
    # If we've loaded the video, show it
    video_path = os.path.join("demo", "videos_full", f"{video_id}.mp4")
    if os.path.exists(video_path):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.video(video_path)
            st.caption(f"Original Video (ID: {video_id})")
    
    # If questions haven't been loaded yet, load them now
    if not st.session_state.questions:
        st.session_state.questions = extract_questions(video_id)
    
    # Question selection section
    st.header("Questions for this video")
    
    # Display existing questions with checkboxes
    if st.session_state.questions:
        st.subheader("Available Questions")
        
        # Create a list to track selected questions
        selected_indices = []
        
        # Display each question with a checkbox
        for i, q in enumerate(st.session_state.questions):
            is_selected = st.checkbox(
                q['text'], 
                key=f"q_{i}_{q.get('qid', 'unknown')}",
                help=f"Type: {q.get('type', 'Unknown')}"
            )
            if is_selected:
                selected_indices.append(i)
        
        # Update selected questions based on checkboxes
        st.session_state.selected_questions = [st.session_state.questions[i] for i in selected_indices]
    else:
        st.info("No predefined questions found for this video ID in the challenge data.")
    
    # # Custom question input
    # st.subheader("Add Custom Question")
    # custom_question = st.text_area(
    #     "Enter your custom question:",
    #     placeholder="Type your question here...",
    #     key="custom_question_input"
    # )
    
    # # Button to add custom question
    # if st.button("Add Custom Question"):
    #     if custom_question.strip():
    #         # Create a new question dictionary
    #         new_question = {
    #             'qid': f"custom_{len(st.session_state.questions)}",
    #             'text': custom_question.strip(),
    #             'type': 'Custom',
    #             'source': 'custom',
    #             'youtube_url': get_video_url_from_id(video_id)
    #         }
            
    #         # Add to questions list
    #         st.session_state.questions.append(new_question)
            
    #         # Show success message
    #         st.success(f"Added custom question: {custom_question.strip()}")
            
    #         # Force a rerun to update the UI
    #         st.rerun()
        
    # Display selected questions count
    st.subheader("Selected Questions")
    if st.session_state.selected_questions:
        st.write(f"You have selected {len(st.session_state.selected_questions)} questions:")
        for q in st.session_state.selected_questions:
            st.write(f"- {q['text']}")
    else:
        st.warning("No questions selected. Please select at least one question for inference.")
    
    # # Button to run inference on selected questions
    # if st.button("Run Inference on Selected Questions", disabled=len(st.session_state.selected_questions) == 0):
    #     if st.session_state.selected_questions:
    #         # Create temporary challenge data file with only selected questions
    #         temp_challenge_path = create_temp_challenge_data(
    #             st.session_state.selected_questions, 
    #             video_id
    #         )
            
    #         # Update the inference command with the temp challenge data path
    #         inference_cmd = steps["Step 3: Run Gemini inference"].format(
    #             challenge_data_path=temp_challenge_path
    #         )
            
    #         # Run the inference step
    #         run_step(inference_cmd, "Step 3: Run Gemini inference", video_id)
            
    #         # Display the formatted output
    #         display_formatted_output(video_id)
    #     else:
    #         st.error("No questions selected. Please select at least one question for inference.")
    
    # Button to run all pipeline steps
    if st.button("Run Full Pipeline", disabled=len(st.session_state.selected_questions) == 0):
        # Run steps 0-2
        for name in ["Step 0: Download metadata", "Step 1: Scrape YouTube comments", "Step 2: Create slowed-down videos"]:
            run_step(steps[name], name, video_id)
        
        # For inference, check if we have selected questions
        if st.session_state.selected_questions:
            # Remove previous inference results for this video ID
            results_path = os.path.join("demo", "results", "demo_gemini_output.csv")
            if os.path.exists(results_path):
                os.remove(results_path)
            
            # Create temporary challenge data file with only selected questions
            temp_challenge_path = create_temp_challenge_data(
                st.session_state.selected_questions, 
                video_id
            )
            
            # Update the inference command with the temp challenge data path
            inference_cmd = steps["Step 3: Run Gemini inference"].format(
                challenge_data_path=temp_challenge_path
            )
            
            # Run the inference step
            run_step(inference_cmd, "Step 3: Run Gemini inference", video_id)
        else:
            # Same for the fallback path...
            results_path = os.path.join("demo", "results", "demo_gemini_output.csv")
            if os.path.exists(results_path):
                os.remove(results_path)
            st.warning("No questions selected for inference. Using all available questions.")
            # Use all questions or fallback to original challenge data
            if st.session_state.questions:
                temp_challenge_path = create_temp_challenge_data(
                    st.session_state.questions, 
                    video_id
                )
                inference_cmd = steps["Step 3: Run Gemini inference"].format(
                    challenge_data_path=temp_challenge_path
                )
            else:
                inference_cmd = steps["Step 3: Run Gemini inference"].format(
                    challenge_data_path="demo/challenge_data.csv"
                )
            
            run_step(inference_cmd, "Step 3: Run Gemini inference", video_id)
        
        # Run step 4
        run_step(steps["Step 4: Submission Format"], "Step 4: Submission Format", video_id)
        
        # Display the formatted output
        display_formatted_output(video_id)

    st.sidebar.header("Run Individual Steps")
    # Sidebar buttons for preprocessing steps
    for name in ["Step 0: Download metadata", "Step 1: Scrape YouTube comments", "Step 2: Create slowed-down videos"]:
        if st.sidebar.button(name):
            run_step(steps[name], name, video_id)
    
    # Special handling for inference step
    if st.sidebar.button("Step 3: Run Gemini inference"):
        # Remove previous inference results
        results_path = os.path.join("demo", "results", "demo_gemini_output.csv")
        if os.path.exists(results_path):
            os.remove(results_path)
        if st.session_state.selected_questions:
            temp_challenge_path = create_temp_challenge_data(
                st.session_state.selected_questions, 
                video_id
            )
            inference_cmd = steps["Step 3: Run Gemini inference"].format(
                challenge_data_path=temp_challenge_path
            )
        else:
            st.sidebar.warning("No questions selected. Using all available questions.")
            if st.session_state.questions:
                temp_challenge_path = create_temp_challenge_data(
                    st.session_state.questions, 
                    video_id
                )
                inference_cmd = steps["Step 3: Run Gemini inference"].format(
                    challenge_data_path=temp_challenge_path
                )
            else:
                inference_cmd = steps["Step 3: Run Gemini inference"].format(
                    challenge_data_path="demo/challenge_data.csv"
                )
        
        run_step(inference_cmd, "Step 3: Run Gemini inference", video_id)
    
    # Step 4 button
    if st.sidebar.button("Step 4: Submission Format"):
        run_step(steps["Step 4: Submission Format"], "Step 4: Submission Format", video_id)
    
    # Button to display formatted output
    if st.sidebar.button("Display Inference Results"):
        display_formatted_output(video_id)
    
    # Always attempt to display formatted output if results exist
    # if os.path.exists(os.path.join("demo", "results", "demo_gemini_output.csv")):
    #     with st.expander("Inference Results", expanded=False):
    #         display_formatted_output(video_id)


if __name__ == "__main__":
    main()