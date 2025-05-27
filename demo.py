import streamlit as st  # UI library for building interactive web apps
import subprocess  # Run shell commands and external processes
import os  # Interact with the operating system


def run_step(command, step_name):
    """
    Execute a shell command and display its live output in the Streamlit app.

    Args:
        command (str): The shell command to execute.
        step_name (str): A descriptive name of the step for UI display.
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
                st.success(f"{step_name} completed successfully.")

                # showing the video
                if step_name == "Step 0: Download metadata":
                    video_path = os.path.join("demo", "videos_full", "HsXS1Qt11cU.mp4")
                    if os.path.exists(video_path):
                        st.video(video_path)
                elif step_name == "Step 2: Create slowed-down videos":
                    video_path = os.path.join(
                        "demo", "videos_full_slow", "HsXS1Qt11cU.mp4"
                    )
                    if os.path.exists(video_path):
                        st.video(video_path)

            else:
                st.error(f"{step_name} failed with exit code {process.returncode}.")
        except Exception as e:
            st.error(f"Error running {step_name}: {e}")


def main():
    """
    Define the Streamlit UI layout and handle user interactions to trigger pipeline steps.
    """
    st.title("AISG NAISC TikTok Challenge")
    st.write(
        "Use the buttons below to run individual steps or execute the entire pipeline sequentially."
    )

    # Map descriptive step names to their shell commands
    steps = {
        "Step 0: Download metadata": (
            "mkdir -p demo && "
            "pushd demo >/dev/null && "
            "python ../demo_download_vids_metadata.py && "
            "popd >/dev/null && "
            "mkdir -p demo/comments demo/videos_full_slow demo/results demo/submissions"
        ),
        "Step 1: Scrape YouTube comments": (
            "python yt_comment_scraper.py "
            "--input demo/video_full_metadata.csv "
            "--output demo/comments "
            "--comments 50 "
            "--verbose"
        ),
        "Step 2: Create slowed-down videos": (
            "python demo_create_slower_vids.py "
            "--input_metadata demo/video_full_metadata.csv "
            "--input_dir demo/videos_full "
            "--output_dir demo/videos_full_slow"
        ),
        "Step 3: Run Gemini inference": (
            "python gemini.py "
            "--challenge_data_path demo/challenge_data.csv "
            "--video_metadata_path demo/video_full_metadata.csv "
            "--comments_dir demo/comments "
            "--output_csv_path demo/results/demo_gemini_output.csv "
            "--model_name models/gemini-2.5-pro-preview-05-06 "
            "--v3 "
            "--temperature 1.0"
        ),
        "Step 4: Submission Format": (
            "python convert_to_upload_format.py "
            "--input demo/results/demo_gemini_output.csv "
            "--output demo/submissions/demo_submission.csv"
        ),
    }

    # Button to run all steps sequentially
    if st.button("Run All Steps"):
        for name, cmd in steps.items():
            run_step(cmd, name)

    st.sidebar.header("Run Individual Steps")
    # Sidebar buttons for each step
    for name, cmd in steps.items():
        if st.sidebar.button(name):
            run_step(cmd, name)


if __name__ == "__main__":
    main()
