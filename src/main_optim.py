import torch
import os
import torch.optim as optim
import torch.nn as nn

from gpt_model import SimpleTransformer
from tokenizer import load_custom_tokenizer
from dataset import TextDataset, prepare_data_loaders
from gpt_train import train_transformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    tokenizer_path="../tokenizers/tokenizer.json",
    dataset_path="../../wikipedia-dump/finalized_exports",
):
    tokenizer = load_custom_tokenizer(tokenizer_path)

    print("Loaded tokenizer.")

    vocab_size = tokenizer.vocab_size

    # Hyperparameters to test
    params = {
        "d_model": [128, 256, 512],
        "nhead": [4, 8, 16],
        "num_layers": [3, 6, 9],
        "dim_feedforward": [1024, 2048, 4096],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.0015, 0.002, 0.005],
    }

    # Find the best combination of hyperparameters
    best_params = None
    best_val_loss = float("inf")
    for d_model in params["d_model"]:
        for nhead in params["nhead"]:
            for num_layers in params["num_layers"]:
                for dim_feedforward in params["dim_feedforward"]:
                    for dropout in params["dropout"]:
                        for learning_rate in params["learning_rate"]:
                            model = SimpleTransformer(
                                vocab_size,
                                d_model,
                                nhead,
                                num_layers,
                                dim_feedforward,
                                dropout,
                            )

                            tokenized_dataset_dir = dataset_path
                            train_batch_size = 32
                            val_batch_size = 32
                            max_seq_len = 200
                            train_dataloader, val_dataloader = prepare_data_loaders(
                                tokenized_dataset_dir,
                                tokenizer,
                                train_batch_size,
                                val_batch_size,
                                max_seq_len,
                            )

                            epochs = 20
                            criterion = nn.CrossEntropyLoss()
                            optimizer = optim.AdamW(
                                model.parameters(), lr=learning_rate, weight_decay=1e-5
                            )
                            device = (
                                "mps" if torch.backends.mps.is_available() else "cpu"
                            )

                            print(
                                "Training model with d_model={}, nhead={}, num_layers={}, dim_feedforward={}, dropout={}, learning_rate={}".format(
                                    d_model,
                                    nhead,
                                    num_layers,
                                    dim_feedforward,
                                    dropout,
                                    learning_rate,
                                )
                            )
                            model.to(device)
                            train_transformer(
                                model,
                                train_dataloader,
                                val_dataloader,
                                vocab_size,
                                epochs,
                                criterion,
                                optimizer,
                                device,
                            )

                            val_loss = evaluate(model, val_dataloader, device)
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_params = {
                                    "d_model": d_model,
                                    "nhead": nhead,
                                    "num_layers": num_layers,
                                    "dim_feedforward": dim_feedforward,
                                    "dropout": dropout,
                                    "learning_rate": learning_rate,
                                }

    print("Best hyperparameters: {}".format(best_params))

    print("Finished training.")


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, output.size(2)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    main(dataset_path="../../wikipedia-dump/export")
