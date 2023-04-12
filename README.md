# reduce-llm

## Contributing

1. Clone the repository with gh cli:
   `gh repo clone benhmoore/reduce-llm`
2. Create a new branch for your addition:
   `git checkout -b [branch-name]`
3. Make your changes.
4. Commit your changes:
   `git commit -m "[commit-message]"`
5. Push your changes:
   `git push --set-upstream origin [branch-name]`

## Helper Scripts

- `chunk_file.py` splits the 16GB wikipedia dump file into smaller files.

  Usage: `python3 chunk_file.py [enwiki-latest-pages-articles.txt] 1000000 -o [output-directory]`
