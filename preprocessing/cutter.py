# This script removes links, title capitalized lines, punctuation, non-standard characters, accented characters, quotation lines and short or empty lines from a text file.
# It also attempts to remove code samples from the text file.

from colorama import Fore, Back, Style
import re
import os
import unicodedata


def process_line(line):
    # Combine multiple spaces into one
    line = re.sub(r'\s+', ' ', line).strip()

    # Remove links
    line = re.sub(r'http\S+', '', line)

    # Replace punctuation
    line = line.replace(';', ',').replace('!', '.')

    # Remove non-standard characters
    line = re.sub(r'[^\x00-\x7F]+', '', line)

    # Remove accented characters
    line = ''.join(c for c in unicodedata.normalize(
        'NFD', line) if unicodedata.category(c) != 'Mn')

    return line


def is_gibberish(line, threshold=0.8):
    if not line:
        return True

    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter_count = sum(1 for char in line if char in letters)
    gibberish_count = sum(1 for char in line if char not in letters)

    if letter_count == 0:
        return True

    gibberish_ratio = gibberish_count / (letter_count + gibberish_count)
    return gibberish_ratio > threshold


def should_remove_line(line):
    # Remove title capitalized lines
    if line.istitle():
        return True

    # Remove code samples
    if re.search(r'[{}<>]', line):
        return True

    # Remove short or empty lines
    if len(line.split()) < 5:
        return True

    # Remove gibberish lines
    if is_gibberish(line):
        return True

    return False


def process_text_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        line = process_line(line)
        if not should_remove_line(line):
            processed_lines.append(line.lower())

    processed_text = '\n'.join(processed_lines)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(processed_text)


def cutter_process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)

        # Process only text files
        if input_file_path.lower().endswith('.txt'):
            process_text_file(input_file_path, output_file_path)
            print(f"Processed {input_file_path} -> {output_file_path}")


if __name__ == "__main__":
    # Get input file path from command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Process a text file and remove links, title capitalized lines, punctuation, non-standard characters, accented characters, quotation lines and short or empty lines.")
    parser.add_argument(
        "input_file", help="Path to the input file.")
    parser.add_argument(
        "output_file", help="Path to the output file.",
        default="output.txt")

    args = parser.parse_args()
    process_text_file(args.input_file, args.output_file)
