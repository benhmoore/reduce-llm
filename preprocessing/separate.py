import re
import os
import argparse

parser = argparse.ArgumentParser(
    description='Condense a text file into a smaller file')
parser.add_argument('file_path', type=str, help='Path to the large text file')
parser.add_argument('-o', '--output_dir', type=str, default='export',
                    help='Output directory for the smaller files (default: export)')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print('Arguments:', args)
print('Starting...')

search_string = "computer"
pat = re.compile(r"\b\b", re.IGNORECASE)  # upper and lowercase will match

mylines = []
with open(args.file_path, 'rt', encoding='utf-8') as myfile:
    for myline in myfile:
        if myline.find(search_string) != -1:
            mylines.append(myline.rstrip('\n'))


def split_file(file_path, output_dir):
    with open(file_path, 'r', encoding='utf-8') as file:
        export_file_path = f'{output_dir}/{os.path.basename(file_path)}_separated.txt'
        current_file = open(export_file_path, 'w', encoding='utf-8')

        for myline in mylines:
            current_file.write(myline)


split_file(args.file_path, args.output_dir)
print(mylines)
