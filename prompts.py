import pandas as pd


def create_comment_relevance_prompt(question, video_title, comments_batch):
    """
    Creates a prompt for evaluating the relevance of YouTube comments to a specific question.

    Parameters:
    -----------
    question : str
        The question we want to answer about the video
    video_title : str
        The title of the video
    comments_batch : list
        A list of comments to evaluate (recommended: 5 comments per batch)

    Returns:
    --------
    str
        A formatted prompt for an LLM to evaluate comment relevance
    """

    # Format the list of comments with numbers
    formatted_comments = "\n".join(
        [f'{i+1}. "{comment}"' for i, comment in enumerate(comments_batch)]
    )

    prompt = f"""You are analyzing YouTube comments to identify those most relevant to answering a specific question about a video.

VIDEO: "{video_title}"
QUESTION: "{question}"

Evaluate each comment's relevance to answering this specific question on a scale of 0-5:

0: Not relevant at all - Provides no useful information for answering the question
1: Minimally relevant - Contains very indirect references that might provide minimal context
2: Somewhat relevant - Contains general information about the video topic but not directly answering the question
3: Moderately relevant - Contains partial information that could help answer the question
4: Highly relevant - Contains important information that directly contributes to answering the question
5: Extremely relevant - Provides critical information that essentially answers the question completely

Comments to evaluate:
{formatted_comments}

For each comment, respond ONLY in this exact format:
Comment #[number]: [score]/5 | [1-2 sentence justification]

Do not include any other text in your response. Be critical and precise in your evaluations."""

    return prompt


def create_video_qa_prompt(
    video_title,
    video_description=None,
    comments=None,
    captions=None,
    channel_name=None,
    question=None,
):
    """
    Constructs a prompt for a video question-answering specialist with enhanced context.

    Parameters:
    -----------
    video_title : str
        The title of the video
    video_description : str, optional
        The description of the video
    comments : list of dict, optional
        List of comment dictionaries with keys 'Author' and 'Comment'
    captions : str, optional
        Video captions in SRT format or as plain text
    channel_name : str, optional
        Name of the YouTube channel that posted the video
    question : str, optional
        The question to be answered about the video

    Returns:
    --------
    str
        The formatted prompt that gracefully handles missing data
    """

    # Start building the prompt
    prompt = "You are a video question-answer specialist. You are provided context from a YouTube video. "
    prompt += "Concisely answer the question at the end.\n\n"

    # Always include the title since it's required
    prompt += "VIDEO INFORMATION:\n"
    prompt += f"Title: {video_title}\n"

    # Add channel name if available
    if channel_name:
        prompt += f"Channel: {channel_name}\n"

    # Add description if available
    if video_description:
        prompt += "\nDESCRIPTION:\n"
        prompt += f"{video_description}\n"

    # Process and add captions if available
    if captions and isinstance(captions, str):  # Check if captions is a valid string
        # Check if captions look like SRT format (contains --> timestamp markers)
        if "-->" in captions:
            cleaned_captions = process_srt_captions(captions)
            prompt += "\nCAPTIONS (TRANSCRIPT):\n"
            prompt += cleaned_captions + "\n"
        else:
            # Treat as plain text captions
            prompt += "\nCAPTIONS (TRANSCRIPT):\n"
            prompt += captions + "\n"
    elif captions is None or pd.isna(captions):  # If captions are missing (None or NaN)
        prompt += "\nCAPTIONS (TRANSCRIPT):\n"
        prompt += "No captions available.\n"

    # Process and add comments if available
    if comments and len(comments) > 0:
        prompt += "\nTOP COMMENTS:\n"
        formatted_comments = format_comments(comments)
        prompt += formatted_comments + "\n"

    # Add the question if available
    if question:
        prompt += "\nQUESTION:\n"
        prompt += f"{question}\n"

    prompt += "\nCONCISE ANSWER:"
    return prompt


def process_srt_captions(srt_text):
    """
    Process SRT formatted captions into a more readable format.

    Parameters:
    -----------
    srt_text : str
        Raw SRT formatted caption text

    Returns:
    --------
    str
        Cleaned caption text with timestamps removed
    """
    import re

    # Split the SRT file into individual subtitle entries
    subtitle_blocks = re.split(r"\n\n+", srt_text.strip())

    cleaned_lines = []

    for block in subtitle_blocks:
        lines = block.split("\n")

        # Skip blocks with fewer than 3 lines (invalid SRT blocks)
        if len(lines) < 3:
            continue

        # Skip the first line (subtitle number) and the second line (timestamp)
        content_lines = lines[2:]

        # Join the remaining lines
        text = " ".join(content_lines)

        # Remove any HTML-style tags
        text = re.sub(r"<[^>]+>", "", text)

        if text:
            cleaned_lines.append(text)

    # Join all the subtitle text with spaces
    return " ".join(cleaned_lines)


def format_comments(comments, max_comments=50):
    """
    Format comments in a more natural and readable way.

    Parameters:
    -----------
    comments : list of dict
        List of comment dictionaries with keys 'Author' and 'Comment'
    max_comments : int, optional
        Maximum number of comments to include

    Returns:
    --------
    str
        Formatted comments text
    """
    formatted_comments = []

    # Limit the number of comments
    comment_subset = comments[:max_comments]

    for i, comment_data in enumerate(comment_subset):
        # Handle both dictionary format and string format
        if isinstance(comment_data, dict):
            author = comment_data.get("Author", "Anonymous User")
            comment_text = comment_data.get("Comment", "")

            # Clean up the comment text
            comment_text = comment_text.replace("\n", " ")

            # Add some basic formatting
            formatted_comment = f"{author}: {comment_text}"
            formatted_comments.append(formatted_comment)
        else:
            # Handle case where comments are provided as strings
            formatted_comments.append(f"User: {comment_data}")

    return "\n".join(formatted_comments)


def create_video_qa_prompt_v2(
    video_title,
    video_description=None,
    comments=None,
    captions=None,
    channel_name=None,
    question=None,
):
    """
    Constructs a prompt for a video question-answering specialist with enhanced context.

    Parameters:
    -----------
    video_title : str
        The title of the video
    video_description : str, optional
        The description of the video
    comments : list of dict, optional
        List of comment dictionaries with keys 'Author' and 'Comment'
    captions : str, optional
        Video captions in SRT format or as plain text
    channel_name : str, optional
        Name of the YouTube channel that posted the video
    question : str, optional
        The question to be answered about the video

    Returns:
    --------
    str
        The formatted prompt that gracefully handles missing data
    """

    # Start building the prompt
    prompt = "You are a video question-answer specialist. You are provided context from a YouTube video. "
    prompt += "Answer the question at the end. Do not output any timestamps."
    prompt += (
        "Think carefully step-by-step and put your thoughts under 'THINKING STEPS'. "
    )
    prompt += "After considering all information, provide a final answer under 'FINAL ANSWER'. Respond by filling up the section after 'RESPONSE TEMPLATE'\n\n"
    # Always include the title since it's required
    prompt += "VIDEO INFORMATION:\n"
    prompt += f"Title: {video_title}\n"

    # Add channel name if available
    if channel_name:
        prompt += f"Channel: {channel_name}\n"

    # Add description if available
    if video_description:
        prompt += "\nDESCRIPTION:\n"
        prompt += f"{video_description}\n"

    # Process and add captions if available
    if captions and isinstance(captions, str):  # Check if captions is a valid string
        # Check if captions look like SRT format (contains --> timestamp markers)
        if "-->" in captions:
            cleaned_captions = process_srt_captions(captions)
            prompt += "\nCAPTIONS (TRANSCRIPT):\n"
            prompt += cleaned_captions + "\n"
        else:
            # Treat as plain text captions
            prompt += "\nCAPTIONS (TRANSCRIPT):\n"
            prompt += captions + "\n"
    elif captions is None or pd.isna(captions):  # If captions are missing (None or NaN)
        prompt += ""

    # Process and add comments if available
    if comments and len(comments) > 0:
        prompt += "\nTOP COMMENTS:\n"
        formatted_comments = format_comments(comments)
        prompt += formatted_comments + "\n"

    # Add the question if available
    if question:
        prompt += "\nQUESTION:\n"
        prompt += f"{question}\n"

    prompt += "\n\nRESPONSE TEMPLATE: "
    prompt += "\nTHINKING STEPS: "
    prompt += "\n\nFINAL ANSWER: "
    return prompt


def create_video_qa_prompt_v3(
    video_title,
    video_description=None,
    comments=None,
    captions=None,
    channel_name=None,
    question=None,
    caution_prompt=False,
):
    """
    Constructs a prompt for a video question-answering specialist with enhanced video descriptions.

    Parameters:
    -----------
    video_title : str
        The title of the video
    video_description : str, optional
        The description of the video
    comments : list of dict, optional
        List of comment dictionaries with keys 'Author' and 'Comment'
    captions : str, optional
        Video captions in SRT format or as plain text
    channel_name : str, optional
        Name of the YouTube channel that posted the video
    question : str, optional
        The question to be answered about the video

    Returns:
    --------
    str
        The formatted prompt that gracefully handles missing data
    """

    # Start building the prompt
    prompt = "You are a video question-answer specialist. You are provided context from a YouTube video. "
    prompt += "Answer the question at the end. Do not output any timestamps."
    prompt += "First, describe in detail what happens at the start, middle, and end of the video. "
    prompt += "Then, think carefully step-by-step and put your thoughts under 'THINKING STEPS'. "
    prompt += "After considering all information, provide a final answer under 'FINAL ANSWER'. Respond by filling up the sections in the response template."
    if caution_prompt:
        prompt += "Be careful: questions and videos may be tricky. Resist misleading assumptions and do not always default to the most obvious answer. Reason about social behavior, context and other human factors related to popular social media, internet trends, an online engagement."
    prompt += "\n\n"
    # Always include the title since it's required
    prompt += "VIDEO INFORMATION:\n"
    prompt += f"Title: {video_title}\n"

    # Add channel name if available
    if channel_name:
        prompt += f"Channel: {channel_name}\n"

    # Add description if available
    if video_description:
        prompt += "\nDESCRIPTION:\n"
        prompt += f"{video_description}\n"

    # Process and add captions if available
    if captions and isinstance(captions, str):  # Check if captions is a valid string
        # Check if captions look like SRT format (contains --> timestamp markers)
        if "-->" in captions:
            cleaned_captions = process_srt_captions(captions)
            prompt += "\nCAPTIONS (TRANSCRIPT):\n"
            prompt += cleaned_captions + "\n"
        else:
            # Treat as plain text captions
            prompt += "\nCAPTIONS (TRANSCRIPT):\n"
            prompt += captions + "\n"
    elif captions is None or pd.isna(captions):  # If captions are missing (None or NaN)
        prompt += ""

    # Process and add comments if available
    if comments and len(comments) > 0:
        prompt += "\nTOP COMMENTS:\n"
        formatted_comments = format_comments(comments)
        prompt += formatted_comments + "\n"

    # Add the question if available
    if question:
        prompt += "\nQUESTION:\n"
        prompt += f"{question}\n"

    prompt += "\n\nRESPONSE TEMPLATE: "
    prompt += "\nVIDEO START DESCRIPTION: "
    prompt += "\n\nVIDEO MIDDLE DESCRIPTION: "
    prompt += "\n\nVIDEO END DESCRIPTION: "
    prompt += "\n\nTHINKING STEPS: "
    prompt += "\n\nFINAL ANSWER: "
    return prompt


def divide_and_conquer_detailed_prompt():
    detailed_prompt = """Please watch the entire video carefully and produce a thorough analysis that covers each of the following key capabilities:

    Plot Attributes (Montage)
        Summarize the overall narrative or storyline.
        Describe how the video is structured (for example, cuts, transitions, or montage sequences).

    Element Counting
        Count and list the main objects, characters, or elements that appear in the video.
        Element Attributes (Optical Illusion)
        Identify and describe any optical illusions, illusions, or visually deceptive techniques.
        Discuss color, shape, or other distinguishing features of significant elements.

    Objective Causality (Videography Phenomenon & Illusion)
        Explain cause-and-effect relationships in the video.
        Include any special filming techniques, illusions, or phenomena that impact viewer perception.

    Professional Knowledge
        Apply relevant domain expertise (e.g., cinematography, science, history, or other specialized knowledge) that helps interpret the events or visuals.

    Event Counting
        Count the main events or actions.
        Provide a brief chronological order.

    Character Motivation Causality
        Discuss the characters’ motivations and why they behave as they do.
        Explain how these motivations drive the storyline.

    Additional Plot Attributes
        Beyond montage, mention any extra narrative elements (plot twists, foreshadowing, pacing, or subplots).

    Event Localization
        Indicate the spatiotemporal location (where and when) for each important event in the video timeline.

    Local Event Attribute
        Describe details unique to each event, such as immediate triggers, changes in environment, or emotional tone.

    Element Localization
        Note where key objects or elements appear in each scene (foreground, background, left, right, etc.).

    Element Attributes
        Elaborate on the appearance and behavior of important objects or elements.
        Discuss materials, colors, or any distinctive properties.

    Positional Relationship
        Explain how characters or objects are arranged relative to each other.
        Include any meaningful shifts in positioning.

    Character Reaction Causality
        Describe how characters react to events or other characters.
        Explain the immediate reasons behind these reactions.
    
    Event Duration & Speed Attribute
        Estimate how long each key event lasts.
        Mention whether any segments are sped up, slowed down, or shown in real-time.

    Character Emotion Attribute
        Identify the emotional states of the characters (e.g., joy, anger, fear).
        Reference facial expressions, body language, or other clues.

    Displacement Attribute
        Note if objects or characters move significantly between locations.
        Describe any changes to their environment or position.
    """
    return detailed_prompt


def divide_and_conquer_default_prompt():
    default_prompt = """Please watch the entire video carefully and produce a thorough analysis of the video. Give as much details as possible about the video content, structure, and any other relevant information.
    
    Note that this is a short video on Youtube. These short videos are likely to produce humorous content.
    """
    return default_prompt
