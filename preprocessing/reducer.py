import os
import glob
import time
import argparse
import json

from utils import *


def reducer_process_directory(input_dir, output_dir, keywords, max_word_distance=5000, max_file_size=104857600, min_chunk_size=5000):
    os.makedirs(output_dir, exist_ok=True)

    file_list = glob.glob(os.path.join(input_dir, "*"))

    output_file_index = 1
    output_file_size = 0
    output_file = open(os.path.join(
        output_dir, f"relevant_export_{output_file_index}.txt"), "w")

    print(f"Processing {len(file_list)} files in '{input_dir}'...")
    print("Looking for the following keywords and sequences:", keywords)

    start_time = time.time()
    total_files = len(file_list)
    time_elapsed_list = []

    for i, file_path in enumerate(file_list, 1):
        file_start_time = time.time()
        print(f"Processing file {i}/{total_files}: {file_path}")

        with open(file_path, "r") as input_file:
            text = input_file.read()
            text = text.lower()
            chunks = chunk_text(text, keywords, max_word_distance)
            for chunk in chunks:
                chunk_size = len(chunk.encode("utf-8"))

                if chunk_size < min_chunk_size:  # Skip chunks that are too small
                    continue

                if output_file_size + chunk_size > max_file_size:
                    output_file.close()
                    output_file_index += 1
                    output_file = open(os.path.join(
                        output_dir, f"relevant_export_{output_file_index}.txt"), "w")
                    output_file_size = 0
                output_file.write(chunk)
                output_file.write("\n\n")
                output_file_size += chunk_size

        file_end_time = time.time()
        time_elapsed = file_end_time - file_start_time
        time_elapsed_list.append(time_elapsed)

        avg_time_per_file = sum(time_elapsed_list) / len(time_elapsed_list)
        remaining_files = total_files - i
        remaining_time = remaining_files * avg_time_per_file

        print(
            f"File {i} processed in {time_elapsed:.2f} seconds. Estimated time remaining: {remaining_time:.2f} seconds.")

    output_file.close()
    total_time_elapsed = time.time() - start_time
    print(
        f"Processing complete. Total time: {total_time_elapsed:.2f} seconds.")

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

    reducer_process_directory(args.input_dir, args.output_dir, keywords,
                              args.max_word_distance, args.max_file_size)
