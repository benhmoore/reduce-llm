from tokenizers import Tokenizer, trainers, pre_tokenizers, models
from transformers import PreTrainedTokenizerFast

def train_tokenizer(data_directory, vocab_size=30000, min_frequency=2, tokenizer_path="tokenizers/custom_tokenizer.json"):
    """
    Train a tokenizer on the dataset and save it to disk.

    Args:
        data_directory (str): The path to the directory containing the text files.
        vocab_size (int): The desired vocabulary size for the tokenizer. Default is 30000.
        min_frequency (int): The minimum frequency for a token to be included in the vocabulary. Default is 2.
        tokenizer_path (str): The path to save the trained tokenizer. Default is "tokenizers/custom_tokenizer.json".
    """

    input_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"])
    tokenizer.train(files=input_files, trainer=trainer)

    tokenizer.save(tokenizer_path)

def load_custom_tokenizer(tokenizer_path="tokenizers/custom_tokenizer.json"):
    """
    Load a custom tokenizer from disk.

    Args:
        tokenizer_path (str): The path to the saved tokenizer. Default is "tokenizers/custom_tokenizer.json".
    
    Returns:
        tokenizer (PreTrainedTokenizerFast): The loaded custom tokenizer.
    """

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<pad>"
    return tokenizer
