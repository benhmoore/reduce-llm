import os
import torch
from gpt_model import SimpleTransformer
from tokenizer import load_custom_tokenizer

# Load the model's state dict from the .pt file


def load_model(model_path, model_class, device):
    model = model_class(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Generate text given a sequence of words


def generate_text(prompt, model, tokenizer, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(input_ids, max_length=max_length, do_sample=True)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


def interactive_shell(model, tokenizer):
    while True:
        prompt = input("Enter a prompt: ")
        generated_text = generate_text(prompt, model, tokenizer)
        print(generated_text)

if __name__ == "__main__":
    # Load the custom tokenizer
    tokenizer_path = "../tokenizers/tokenizer.json"
    tokenizer = load_custom_tokenizer(tokenizer_path)

    # Set model parameters
    vocab_size = tokenizer.vocab_size
    d_model = 128
    nhead = 4
    num_layers = 3
    dim_feedforward = 1024
    model_path = "../trained_models/epoch_1_76989.pt"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load the model
    model = load_model(model_path, SimpleTransformer, device)
    model.to(device)


    # Launch interactive shell
    interactive_shell(model, tokenizer)

    # # Generate text
    # prompt = "The quick brown fox jumps over the lazy dog "
    # generated_text = generate_text(prompt, model, tokenizer)
    # print(generated_text)
