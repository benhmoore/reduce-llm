# This script creates a tokenizer using Hugging Face's tokenizers library. It uses the BPE model to tokenize the text files in the processed-export directory.
# Be sure to set the path to your text files directory.
from tokenizers import Tokenizer, trainers, pre_tokenizers, models
import os

# Set the path to your text files directory
data_directory = "../wikipedia-dump/finalized_exports"
input_files = [os.path.join(data_directory, f)
               for f in os.listdir(data_directory)]

# Initialize a new tokenizer with the desired tokenization model (BPE in this case)
tokenizer = Tokenizer(models.BPE())

# Set a pre-tokenizer to split the text into words
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train the tokenizer on your dataset
trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=[
                              "<pad>", "<s>", "</s>", "<unk>", "<mask>"])
tokenizer.train(files=input_files, trainer=trainer)

# Save the trained tokenizer to disk
tokenizer.save("tokenizer.json")
