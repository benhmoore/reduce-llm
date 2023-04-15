import os
import glob
import argparse
import json

from utils import *


def process_directory(input_dir, output_dir, keywords, max_word_distance, max_file_size):
    os.makedirs(output_dir, exist_ok=True)

    file_list = glob.glob(os.path.join(input_dir, "*"))

    output_file_index = 1
    output_file_size = 0
    output_file = open(os.path.join(
        output_dir, f"relevant_export_{output_file_index}.txt"), "w")

    print(f"Processing {len(file_list)} files in '{input_dir}'...")
    print("Looking for the following keywords and sequences:", keywords)

    for i, file_path in enumerate(file_list, 1):
        print(f"Processing file {i}/{len(file_list)}: {file_path}")

        with open(file_path, "r") as input_file:
            text = input_file.read()
            chunks = chunk_text(text, keywords, max_word_distance)
            for chunk in chunks:
                chunk_size = len(chunk.encode("utf-8"))
                if output_file_size + chunk_size > max_file_size:
                    output_file.close()
                    output_file_index += 1
                    output_file = open(os.path.join(
                        output_dir, f"relevant_export_{output_file_index}.txt"), "w")
                    output_file_size = 0
                output_file.write(chunk)
                output_file.write("\n\n")
                output_file_size += chunk_size

    output_file.close()
    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a directory of files and extract relevant chunks based on keywords.")
    parser.add_argument(
        "input_dir", help="Path to the directory containing input files.")
    parser.add_argument(
        "output_dir", help="Path to the directory where output files will be saved.")
    parser.add_argument(
        "keywords_file", help="Path to the JSON file containing keywords and ordered sequences.")
    parser.add_argument("max_word_distance", type=int,
                        help="Maximum word distance between keywords in a cluster.")
    parser.add_argument("max_file_size", type=int,
                        help="Maximum size of output files in bytes.")

    args = parser.parse_args()

    with open(args.keywords_file, "r") as f:
        keywords = json.load(f)

    process_directory(args.input_dir, args.output_dir, keywords,
                      args.max_word_distance, args.max_file_size)

# Example Usage
# python helper-scripts/reducer/reduce.py ../wikipedia-dump/export ../wikipedia-dump/reduced-export helper-scripts/reducer/example_keywords.json 1000 104857600
