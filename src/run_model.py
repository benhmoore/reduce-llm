import os
import torch
from gpt_model import SimpleTransformer
from tokenizer import load_custom_tokenizer
from colorama import Fore, Style

import random

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer

from colorama import Fore, Style

nltk.download("punkt")
nltk.download("words")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("vader_lexicon")

from nltk.corpus import words


def count_misspelled_words(text):
    word_list = words.words()
    words_set = set(word_list)
    tokens = nltk.word_tokenize(text)
    misspelled_tokens = [token for token in tokens if token.lower() not in words_set]
    return len(misspelled_tokens)


def is_misspelled(word, word_list):
    return word.lower() not in word_list


# Load the model's state dict from the .pt file
def load_model(model_path, model_class, device):
    model = model_class(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Generate text given a sequence of words
def generate_text(input_ids, model, tokenizer, max_length=30):
    generated = model.generate(input_ids, max_length=max_length, do_sample=True)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


def generate_ten_responses(prompt, model, tokenizer):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    responses = []
    for _ in range(40):
        generated_text = generate_text(input_ids, model, tokenizer)
        responses.append(generated_text)
    print("Generated responses:", "\n\n".join(responses), sep="\n")
    return responses


def compute_score(response):
    # Heavily penalize empty sentences or those containing only whitespace
    if not response or response.isspace():
        return -1000

    tokenized = word_tokenize(response)
    tagged = pos_tag(tokenized)

    # Penalize short sentences (less than 5 words)
    if len(tokenized) < 5:
        return 0

    # Reward longer sentences
    length_score = len(tokenized) / 10

    # Reward proper use of nouns and verbs
    pos_score = sum(
        1 for word, tag in tagged if tag.startswith("N") or tag.startswith("V")
    )

    # Reward diverse vocabulary
    unique_words = len(set(tokenized)) / len(tokenized)
    vocab_score = unique_words * 10

    # Reward semantically coherent sentences
    synsets = [wn.synsets(token) for token in tokenized if wn.synsets(token)]
    coherence_score = sum(len(synset) for synset in synsets) / len(synsets)

    # # Penalize negative sentiment
    # sia = SentimentIntensityAnalyzer()
    # sentiment_score = sia.polarity_scores(response)["compound"]

    # Penalize misspellings using nltk.corpus.words
    word_list = set(words.words())
    misspellings_penalty = sum(is_misspelled(word, word_list) for word in tokenized) * 2

    # Compute the final score
    final_score = (
        length_score + pos_score + vocab_score + coherence_score - misspellings_penalty
    )

    return final_score


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
