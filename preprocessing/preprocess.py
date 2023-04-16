from chunk_file import *
from cutter import *
from reducer import *
from colorama import Fore, Back, Style


def process_directory(input_dir, output_dir, keywords_file):

    print(Back.GREEN, "Running rEDUCER", Style.RESET_ALL)
    print(" * * * * * * * ")
    print("Attempting to remove irrelevant chunks and combine chunks into files of a specified size.")
    print(" * * * * * * * ")

    print(Fore.CYAN, "Loading keywords file... ",
          keywords_file, Style.RESET_ALL)
    with open(keywords_file, "r") as f:
        keywords = json.load(f)
        print(Fore.GREEN, "Loaded. Using keywords: ", keywords, Style.RESET_ALL)

    reducer_process_directory(input_dir, "reducer_output", keywords)

    print(Back.GREEN, "Running cUTTER", Style.RESET_ALL)
    print(" * * * * * * * ")
    print("Attempting to remove links, titles and headings, non-standard characters, code, quotations, and gibberish.")
    print(" * * * * * * * ")
    cutter_process_directory("reducer_output", output_dir)

    print(Back.GREEN, "Cut complete", Style.RESET_ALL)

    print("Check the output directory for the processed files.", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a directory of files and extract relevant chunks based on keywords.")
    parser.add_argument(
        "input_dir", help="Path to the directory containing input files.")
    parser.add_argument(
        "output_dir", help="Path to the directory where output files will be saved.")
    parser.add_argument(
        "keywords_file", help="Path to the JSON file containing keywords and ordered sequences.")

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.keywords_file)
