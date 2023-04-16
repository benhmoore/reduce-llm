import os
import pickle
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split


class TextDataset(Dataset):
    """
    A custom PyTorch Dataset class for handling tokenized text sequences.

    Args:
        input_sequences (torch.Tensor): Input sequences tensor of shape (n_samples, seq_length).
        target_sequences (torch.Tensor): Target sequences tensor of shape (n_samples, seq_length).
    """

    def __init__(self, input_sequences, target_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]


def create_dataset(tokenizer, max_seq_len):
    """
    Create a dataset from a directory of text files.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer object.
        max_seq_len (int): The maximum sequence length for input/target sequences.

    Returns:
        dataset (Dataset): The created TextDataset object.
    """

    # Replace with the path to your raw data directory
    data_dir = "../../wikipedia-dump/finalized_exports"
    input_sequences = []
    target_sequences = []

    for file_path in Path(data_dir).glob("**/*.txt"):
        print(f"Processing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            tokens = tokenizer.encode(text)  # Encode the text into tokens

            # Create input and target pairs
            # The target sequence is the input sequence shifted by 1
            for i in range(0, len(tokens) - max_seq_len, max_seq_len):
                input_sequence = tokens[i:i + max_seq_len]
                target_sequence = tokens[i + 1:i + max_seq_len + 1]
                input_sequences.append(torch.tensor(
                    input_sequence, dtype=torch.long))
                target_sequences.append(torch.tensor(
                    target_sequence, dtype=torch.long))

    input_sequences = torch.stack(input_sequences)
    target_sequences = torch.stack(target_sequences)
    dataset = TextDataset(input_sequences, target_sequences)

    return dataset


def save_dataset(dataset, file_path):
    """
    Save a dataset object to a file.

    Args:
        dataset (Dataset): The dataset object to be saved.
        file_path (str): The path to save the dataset file.
    """

    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved dataset to {file_path}.")


def load_dataset(file_path):
    """
    Load a dataset object from a file.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        dataset (Dataset): The loaded dataset object.
    """

    with open(file_path, "rb") as f:
        return pickle.load(f)

    print(f"Loaded dataset from {file_path}.")


def prepare_data_loaders(directory, tokenizer, train_batch_size=32, val_batch_size=32, max_seq_len=200, save_to_disk=True):
    """
    Prepare DataLoader objects for training and validation datasets.

    Args:
        directory (str): The path to the directory containing the tokenized dataset file.
        tokenizer (PreTrainedTokenizerFast): The tokenizer object.
        train_batch_size (int): The batch size for the training DataLoader. Default is 32.
        val_batch_size (int): The batch size for the validation DataLoader. Default is 32.
        max_seq_len (int): The maximum sequence length for input/target sequences. Default is 200.
        save_to_disk (bool): Whether to save the dataset object to disk. Default is True.

    Returns:
        train_dataloader (DataLoader): DataLoader object for the training dataset.
        val_dataloader (DataLoader): DataLoader object for the validation dataset.
    """

    # Load the tokenized dataset
    dataset_file = os.path.join(directory, f'dataset_{max_seq_len}.pkl')
    if os.path.exists(dataset_file):
        print("Loading existing dataset...")
        dataset = load_dataset(dataset_file)
    else:
        print("Creating dataset...")
        dataset = create_dataset(tokenizer, max_seq_len)
        if save_to_disk:
            save_dataset(dataset, dataset_file)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Prepare the data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader
