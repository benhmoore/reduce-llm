# Build array of lines from file, strip newlines
import re
import os
import argparse


parser = argparse.ArgumentParser(
    description='Condense a text file into a smaller file')
filename = parser.add_argument('file_path', type=str, help='Path to the large text file')
#parser.add_argument('lines_per_file', type=int, default=1000000,
 #                   help='Number of lines per output file')
parser.add_argument('-o', '--output_dir', type=str, default='export',
                    help='Output directory for the smaller files (default: export)')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Print the arguments and start message
print('Arguments:', args)
print('Starting...')



filename = input("Enter a filename: ")

str = "computer"
pat = re.compile(r"\b\b", re.IGNORECASE)  # upper and lowercase will match

mylines = []                                # Declare an empty list.
with open (filename, 'rt', encoding='utf-8') as myfile:    # Open lorem.txt for reading text.
    for myline in myfile:                   # For each line in the file,
        if myline.find(str) != -1:
            mylines.append(myline.rstrip('\n'))
        #if pat.search(str) != None:
            #print("Found it.")
        #    mylines.append(myline.rstrip('\n')) # strip newline and add to list.

def split_file(file_path, output_dir):
    with open(file_path, 'r', encoding='utf-8') as file:
        export_file_path = f'{output_dir}/{filename}_separated.txt'
        current_file = open(export_file_path, 'w', encoding='utf-8')

        for myline in mylines:
            current_file.write(myline)

#parser = argparse.ArgumentParser(
#    description='Condense a text file into a smaller file')
#filename = parser.add_argument('file_path', type=str, help='Path to the large text file')

##parser.add_argument('lines_per_file', type=int, default=1000000,
 ##                   help='Number of lines per output file')

#parser.add_argument('-o', '--output_dir', type=str, default='export',
#                    help='Output directory for the smaller files (default: export)')

#args = parser.parse_args()

#if not os.path.exists(args.output_dir):
#    os.makedirs(args.output_dir)

split_file(args.file_path, args.output_dir)
print(mylines)





