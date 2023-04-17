import torch
import torch.optim as optim
import torch.nn as nn
import time


def train_transformer(model, train_dataloader, val_dataloader, vocab_size, epochs, criterion, optimizer, device):
    # Training loop
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        model.train()
        total_loss = 0

        batch_count = 0
        for input_sequences, target_sequences in train_dataloader:
            batch_count += 1
            if batch_count % 50 == 0:
                print(
                    f"Processing batch {batch_count} of {len(train_dataloader)}")
                start_time = time.time()

            input_sequences = input_sequences.to(device)
            target_sequences = target_sequences.to(device)

            optimizer.zero_grad()
            output = model(input_sequences)

            loss = criterion(
                output.view(-1, vocab_size), target_sequences.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print loss for every batch
            print(f"Batch: {batch_count}, Loss: {loss.item():.4f}")

            if batch_count % 50 == 0:
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
                    output.view(-1, vocab_size), target_sequences.view(-1))
                total_loss += loss.item()

            avg_loss = total_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_loss:.4f}")

        # Save the model after each epoch to the ../trained_models directory
        model_path = f"../trained_models/epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), model_path)
