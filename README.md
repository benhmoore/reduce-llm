# reduce-llm

## Contributing

1. Clone the repository with gh cli (https://cli.github.com):
   `gh repo clone benhmoore/reduce-llm`
2. Create a new branch for your addition:
   `git checkout -b [branch-name]`
3. Make your changes.
4. Commit your changes:
   `git commit -m "[commit-message]"`
5. Push your changes:
   `git push --set-upstream origin [branch-name]`

## Installing the environment

This project requires Python 3.9 and Conda to run. You can install Conda by following the instructions on https://docs.conda.io/en/latest/miniconda.html.

To create a conda environment with all the necessary packages for this project, you can use the environment.yml file provided in this repository. Follow these steps:

1. Open a terminal and navigate to the folder where environment.yml is stored.
2. Run the command `conda env create --file environment.yml`. This will create a new environment called `reduce-llm` with the specified channels and dependencies.
3. To activate the environment, run the command `conda activate reduce-llm`.
4. To verify that the environment was created successfully, run the command `conda list` and check that the packages are installed.

You are now ready to run the project code in the `reduce-llm` environment.

## Steps for Training

1. Split the Wikipedia dump file into smaller files using `chunk_file.py`.
2. Preprocess the files using `preprocess.py`.
3. Generate a tokenizer using `generate_tokenizer.py`.
   4...

## Helpers

- `chunk_file.py` splits the 16GB wikipedia dump file into smaller files.

  Usage: `python3 chunk_file.py [enwiki-latest-pages-articles.txt] 1000000 -o [output-directory]`

### Preprocess

The `preprocess.py` CLI tool is used to preprocess the files exported by chunk_file.py. It does this by chunking the files based on the occurrence of keywords using DBSCAN clustering and removing irrelevant text.

Syntax:

```bash
python preprocessing/preprocess.py [input-chunk-dir] [process-output-dir] [keywords-file.json]
```

Example usage:

```bash
python preprocessing/preprocess.py ../wikipedia-dump/export finalized_exports preprocessor_keywords.json
```

### Reduce

This CLI tool is used to reduce the size of the files exported by chunk_file.py. It is used as a component of `preprocess.py`. It does this by chunking the files based on the occurrence of keywords using DBSCAN clustering.

The result are files that only contain text related to the keywords.

Syntax:

```bash
python reducer.py [input-export-directory] [output-reduced-export-directory] [keywords-file.json] [chunk-size] [max-file-size]
```

Example usage:

```bash
python preprocessing/reducer.py ../wikipedia-dump/export ../wikipedia-dump/reduced-export preprocessing/example_keywords.json 1000 104857600
```
