import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class TextDataset(Dataset):
    def __init__(self, tokenizer, directory, max_seq_len=200):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = []

        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            with open(file_path, "r", encoding="utf-8") as f:
                self.texts.extend(f.read().splitlines())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=self.max_seq_len)

        # Pad the sequence
        padded_input_ids = input_ids + \
            [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))

        # Create input and target pairs
        input_sequence = torch.tensor(padded_input_ids[:-1])
        target_sequence = torch.tensor(padded_input_ids[1:])

        return input_sequence, target_sequence


def save_dataset(dataset, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def prepare_data_loaders(directory, tokenizer, train_batch_size=32, val_batch_size=32, max_seq_len=200, save_to_disk=True):
    # Load dataset from disk if it exists, otherwise create and save it
    dataset_file = os.path.join(directory, f"dataset_{max_seq_len}.pkl")
    if os.path.exists(dataset_file):
        dataset = load_dataset(dataset_file)
    else:
        dataset = TextDataset(tokenizer, directory, max_seq_len)
        if save_to_disk:
            save_dataset(dataset, dataset_file)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_dataloader, val_dataloader


# Load a pre-trained tokenizer (replace "bert-base-uncased" with your tokenizer of choice)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare data loaders

# Load the trained tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizers/tokenizer.json")

# Add the padding token to the tokenizer to make it compatible with the TextDataset class
tokenizer.pad_token = "<pad>"

# Prepare data loaders
data_directory = "../wikipedia-dump/finalized_exports"
train_dataloader, val_dataloader = prepare_data_loaders(
    data_directory, tokenizer)
