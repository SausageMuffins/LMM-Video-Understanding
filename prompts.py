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
    formatted_comments = "\n".join([f"{i+1}. \"{comment}\"" for i, comment in enumerate(comments_batch)])
    
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
    question=None
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
    subtitle_blocks = re.split(r'\n\n+', srt_text.strip())
    
    cleaned_lines = []
    
    for block in subtitle_blocks:
        lines = block.split('\n')
        
        # Skip blocks with fewer than 3 lines (invalid SRT blocks)
        if len(lines) < 3:
            continue
        
        # Skip the first line (subtitle number) and the second line (timestamp)
        content_lines = lines[2:]
        
        # Join the remaining lines
        text = ' '.join(content_lines)
        
        # Remove any HTML-style tags
        text = re.sub(r'<[^>]+>', '', text)
        
        if text:
            cleaned_lines.append(text)
    
    # Join all the subtitle text with spaces
    return ' '.join(cleaned_lines)


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
            author = comment_data.get('Author', 'Anonymous User')
            comment_text = comment_data.get('Comment', '')
            
            # Clean up the comment text
            comment_text = comment_text.replace('\n', ' ')
            
            # Add some basic formatting
            formatted_comment = f"{author}: {comment_text}"
            formatted_comments.append(formatted_comment)
        else:
            # Handle case where comments are provided as strings
            formatted_comments.append(f"User: {comment_data}")
    
    return '\n'.join(formatted_comments)

def create_video_qa_prompt_v2(
    video_title,
    video_description=None,
    comments=None,
    captions=None,
    channel_name=None,
    question=None
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
    prompt += "Think carefully step-by-step and put your thoughts under 'THINKING STEPS'. "
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