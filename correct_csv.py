import pandas as pd


def correct_video_id_df(input_file, output_file):
    """
    Read the CSV file, update the video_id by extracting the substring after the last '/'
    from the youtube_url, wrap it in double quotes, and write the updated data to a new CSV file.
    """
    # Read CSV file with automatic delimiter detection
    df = pd.read_csv(
        input_file, sep=None, engine="python"
    )  # engine='python' supports regex-based delimiter detection

    # Extract the video id from the youtube_url
    # The .str.split('/') extracts parts of the string, and .str[-1] gets the last segment.
    # Then, wrap it in double quotes.
    df["video_id"] = (
        df["youtube_url"].str.strip().str.split("/").str[-1].apply(lambda x: f'"{x}"')
    )

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)


# Usage example:
input_csv = "challenge_data.csv"
output_csv = "challenge_data_corrected.csv"
correct_video_id_df(input_csv, output_csv)
