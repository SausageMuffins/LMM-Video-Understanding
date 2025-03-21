#!/usr/bin/env python3
import pandas as pd
import argparse

def convert_csv(input_file, output_file, answer_column):
    """
    Convert the input CSV file to the required output format.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        answer_column: Column name to map to 'pred' in the output
    """
    try:
        # Read the input CSV
        df = pd.read_csv(input_file)
        
        # Check if the required columns exist
        if 'qid' not in df.columns:
            raise ValueError(f"Input CSV must contain 'qid' column. Available columns: {df.columns.tolist()}")
        
        if answer_column not in df.columns:
            raise ValueError(f"Column '{answer_column}' not found in the input CSV. Available columns: {df.columns.tolist()}")
        
        # Create the output dataframe with only 'qid' and 'pred' columns
        output_df = pd.DataFrame()
        output_df['qid'] = df['qid']
        output_df['pred'] = df[answer_column]
        
        # Write to output CSV file
        output_df.to_csv(output_file, index=False)
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        print(f"Mapped '{answer_column}' column to 'pred'")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Convert CSV to upload format')
    parser.add_argument('--input', default='challenge_data_baseline.csv', 
                        help='Input CSV file (default: challenge_data_baseline.csv)')
    parser.add_argument('--output', default='pred_baseline.csv', 
                        help='Output CSV file (default: pred_baseline.csv)')
    parser.add_argument('--answer-column', default='answer', 
                        help='Column name to map to "pred" in the output (default: answer)')
    
    args = parser.parse_args()
    
    # Convert the CSV
    convert_csv(args.input, args.output, args.answer_column)

if __name__ == "__main__":
    main()