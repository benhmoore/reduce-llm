# This script splits a large text file into smaller files

import os
import argparse


def split_large_file(file_path, lines_per_file, output_dir):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_count = 0
        current_line_count = 0
        current_file = None

        for line in file:
            if current_line_count == 0:
                if current_file:
                    current_file.close()

                file_count += 1
                export_file_path = f'{output_dir}/part_{file_count:04d}.txt'
                current_file = open(export_file_path, 'w', encoding='utf-8')

            current_file.write(line)
            current_line_count += 1

            if current_line_count == lines_per_file:
                current_line_count = 0

        if current_file:
            current_file.close()


# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split a large text file into smaller files')
    parser.add_argument('file_path', type=str,
                        help='Path to the large text file')
    parser.add_argument('lines_per_file', type=int, default=1000000,
                        help='Number of lines per output file')
    parser.add_argument('-o', '--output_dir', type=str, default='export',
                        help='Output directory for the smaller files (default: export)')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Print the arguments and start message
    print('Arguments:', args)
    print('Starting...')

    split_large_file(args.file_path, args.lines_per_file, args.output_dir)
