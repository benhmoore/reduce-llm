import os

def split_file(file_path, output_dir, chunk_size=50*1024*1024):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    num_chunks = (file_size // chunk_size) + 1

    with open(file_path, 'rb') as infile:
        for i in range(num_chunks):
            chunk_file_name = f"{file_name}.part{i+1}.txt"
            chunk_file_path = os.path.join(output_dir, chunk_file_name)
            with open(chunk_file_path, 'wb') as outfile:
                outfile.write(infile.read(chunk_size))
            print(f"Created chunk: {chunk_file_path}")

if __name__ == "__main__":
    large_file_path = "../wikipedia-dump/cs-articles-utf8/cs-articles.txt"
    output_directory = "export"
    split_file(large_file_path, output_directory)
