import torch
import torch.optim as optim
import torch.nn as nn
import time


def train_transformer(model, train_dataloader, val_dataloader, epochs, criterion, optimizer, device):
    """
    Train a transformer model using the provided data loaders, loss function, and optimizer.

    Args:
        model (nn.Module): The transformer model to train.
        train_dataloader (DataLoader): The DataLoader for the training dataset.
        val_dataloader (DataLoader): The DataLoader for the validation dataset.
        epochs (int): The number of epochs to train the model.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer to use for updating the model parameters.
        device (torch.device): The device to run the model on (e.g., "cuda" or "cpu").
    """

    # Training loop
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        model.train()
        total_loss = 0

        batch_count = 0
        for input_sequences, target_sequences in train_dataloader:
            batch_count += 1
            print(f"Processing batch {batch_count} of {len(train_dataloader)}")
            start_time = time.time()

            input_sequences = input_sequences.to(device)
            target_sequences = target_sequences.to(device)

            optimizer.zero_grad()
            output = model(input_sequences)

            loss = criterion(
                output.view(-1, model.embedding.num_embeddings), target_sequences.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            end_time = time.time()
            time_per_batch = end_time - start_time
            print(f"Time per batch: {time_per_batch} seconds")
            print(
                f"Estimated time to completion: {(len(train_dataloader) - batch_count) * time_per_batch / 3600:.4f} hours")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")

        # Validation loop
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for input_sequences, target_sequences in val_dataloader:
                input_sequences = input_sequences.to(device)
                target_sequences = target_sequences.to(device)

                output = model(input_sequences)
                loss = criterion(
                    output.view(-1, model.embedding.num_embeddings), target_sequences.view(-1))
                total_loss += loss.item()

            avg_loss = total_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_loss:.4f}")

        # Save the model after each epoch to the ../trained_models directory
        model_path = f"../trained_models/epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), model_path)
