import os
import torch
from gpt_model import SimpleTransformer
from tokenizer import load_custom_tokenizer
from colorama import Fore, Style
import nltk

import random
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from colorama import Fore, Style

nltk.download("punkt")
nltk.download("words")
nltk.download("averaged_perceptron_tagger")

from nltk.corpus import words


def count_misspelled_words(text):
    word_list = words.words()
    words_set = set(word_list)
    tokens = nltk.word_tokenize(text)
    misspelled_tokens = [token for token in tokens if token.lower() not in words_set]
    return len(misspelled_tokens)


# Load the model's state dict from the .pt file
def load_model(model_path, model_class, device):
    model = model_class(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Generate text given a sequence of words
def generate_text(input_ids, model, tokenizer, max_length=50):
    generated = model.generate(input_ids, max_length=max_length, do_sample=True)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


def generate_ten_responses(prompt, model, tokenizer):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    responses = []
    for _ in range(10):
        generated_text = generate_text(input_ids, model, tokenizer)
        responses.append(generated_text)
    print("Generated responses:", "\n\n".join(responses), sep="\n")
    return responses


def compute_score(response):
    tokenized = word_tokenize(response)
    tagged = pos_tag(tokenized)

    # You can customize the scoring method based on the POS tags or other criteria.
    score = sum(1 for word, tag in tagged if tag.startswith("N") or tag.startswith("V"))

    return score


def process_generation(generated_text):
    # Capitalize the first letter of the generated text
    capitalized_text = generated_text.capitalize()
    # Find the indices of all sentence-ending punctuation in the text
    sentence_end_indices = []
    for sent in nltk.sent_tokenize(capitalized_text):
        sent_len = len(sent)
        if sent_len > 0 and sent[sent_len - 1] in [".", "?", "!"]:
            sentence_end_indices.append(capitalized_text.index(sent) + sent_len - 1)
    # Capitalize the letter after each sentence-ending punctuation
    for i in sentence_end_indices:
        if i + 2 < len(capitalized_text):
            capitalized_text = (
                capitalized_text[: i + 2] + capitalized_text[i + 2 :].capitalize()
            )
    return capitalized_text


def interactive_shell(model, tokenizer):
    while True:
        prompt = input(Fore.YELLOW + "Enter a prompt: ")
        print(Fore.RESET)
        responses = generate_ten_responses(prompt, model, tokenizer)

        scored_responses = [
            (response, compute_score(response)) for response in responses
        ]
        best_response = max(scored_responses, key=lambda x: x[1])[0]

        print("-" * 80)
        best_response = prompt[0] + process_generation(best_response)

        num_words = len(best_response.split())
        num_misspelled = count_misspelled_words(best_response)
        print("Misspelled word ratio = {:.2f}".format(num_misspelled / num_words))

        # Print parameter count
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters = {:,}".format(num_params))

        # Remove sentence fragment at end of response
        if best_response[-1] not in [".", "?", "!"]:
            best_response = best_response[: best_response.rfind(".") + 1]

        print(Fore.GREEN + best_response + Style.RESET_ALL)
        print("-" * 80)


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
    model_path = "../trained_models/epoch_20_24889_4.631118524899752.pt"

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
