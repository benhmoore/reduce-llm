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
