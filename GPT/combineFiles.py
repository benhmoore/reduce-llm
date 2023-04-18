import os
import glob
path = os.path.join("/mnt/d/Projects/reduce-llm/cs-articles/*.txt")
file_list = glob.glob(path)
output_file = open(os.path.join("/mnt/d/Projects/reduce-llm/cs-articles-combined/cs-articles-combined.txt"), "w")
for file_path in file_list:
    with open(file_path, "r") as input_file:
        text = input_file.read()
        output_file.write(text)
        output_file.write("\n")

output_file.close()
