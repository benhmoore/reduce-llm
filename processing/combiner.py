import os
import chardet

def merge_files(input_directory, output_file_path):
    # List all files in the input directory
    files = os.listdir(input_directory)

    # Filter out directories, only keep files
    files = [file for file in files if os.path.isfile(os.path.join(input_directory, file))]

    for file in files:
        file_path = os.path.join(input_directory, file)

        # Read the raw binary content of the input file
        with open(file_path, 'rb') as input_file:
            raw_data = input_file.read()

        # Detect the encoding of the input file
        detected_encoding = chardet.detect(raw_data)['encoding']

        # Decode the input file content using the detected encoding
        content = raw_data.decode(detected_encoding)

        # Encode the content as UTF-8 and append it to the output file
        with open(output_file_path, 'ab') as output_file:
            output_file.write(content.encode('utf-8'))
            output_file.write(b'\n')


if __name__ == "__main__":
    input_directory = '../wikipedia-dump/cs-articles'
    output_file_path = 'cs-articles.txt'
    merge_files(input_directory, output_file_path)
