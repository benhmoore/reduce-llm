import torch
import os
import torch.optim as optim
import torch.nn as nn

# from model import SimpleTransformer
from gpt_model import SimpleTransformer

from tokenizer import load_custom_tokenizer
from dataset import TextDataset, prepare_data_loaders
from gpt_train import train_transformer

# Disable parallelism for tokenizers to avoid errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(tokenizer_path="../tokenizers/tokenizer.json", dataset_path="../../wikipedia-dump/finalized_exports"):
    # Load the custom tokenizer
    tokenizer = load_custom_tokenizer(tokenizer_path)

    print("Loaded tokenizer.")

    # Create the model
    vocab_size = tokenizer.vocab_size
    d_model = 256  # d_model is the dimension of the encoder/decoder layers
    nhead = 4  # nhead is the number of heads in the multiheadattention models
    num_layers = 6  # num_layers is the number of encoder and decoder layers
    # dim_feedforward is the dimension of the feedforward network model in nn.TransformerEncoder
    dim_feedforward = 1512
    model = SimpleTransformer(
        vocab_size, d_model,
        nhead, num_layers,
        dim_feedforward,
        dropout=0.05,
        # train_x=None,
        # train_y=None
    )

    print("Created model.")

    # Prepare the data loaders
    tokenized_dataset_dir = dataset_path
    train_batch_size = 32
    val_batch_size = 32
    max_seq_len = 200
    train_dataloader, val_dataloader = prepare_data_loaders(
        tokenized_dataset_dir, tokenizer, train_batch_size, val_batch_size, max_seq_len)

    print("Prepared data loaders.")

    # Train the model
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Training model.")
    model.to(device)
    train_transformer(model, train_dataloader, val_dataloader, vocab_size,
                      epochs, criterion, optimizer, device)

    print("Finished training.")


if __name__ == "__main__":
    main(dataset_path="../../wikipedia-dump/preprocessed")
