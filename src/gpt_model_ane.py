import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizer import load_custom_tokenizer


class SimpleTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1
    ):
        super().__init__()

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_head=nhead,
            n_layer=num_layers,
            n_positions=dim_feedforward,
            n_ctx=dim_feedforward,
            use_cache=False,
            dropout=dropout,
        )
        self.gpt = GPT2LMHeadModel(config=gpt_config)

    def forward(self, x):
        x = self.gpt(input_ids=x, return_dict=True).logits
        return x

    def generate(
        self,
        input_ids,
        max_length,
        do_sample=True,
        temperature=1.0,
        num_beams=1,
        top_k=0,
        top_p=1.0,
    ):
        with torch.no_grad():
            generated_text = self.gpt.generate(
                input_ids,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p,
            )
        return generated_text


# # Load the custom tokenizer
# tokenizer_path = "../tokenizers/tokenizer.json"
# tokenizer = load_custom_tokenizer(tokenizer_path)

# input_ids = tokenizer.encode("Hello world", return_tensors="pt")
# example = input_ids  # A tensor of shape [1, 3]

# # Set model parameters
# vocab_size = tokenizer.vocab_size
# d_model = 128
# nhead = 4
# num_layers = 3
# dim_feedforward = 1024
# model_path = "../trained_models/epoch_20_24889_4.631118524899752.pt"

# # Create an instance of your model
# model = SimpleTransformer(
#     vocab_size=vocab_size,
#     d_model=d_model,
#     nhead=nhead,
#     num_layers=3,
#     dim_feedforward=1024,
# )


# # Load the weights from the .pth file
# model.load_state_dict(
#     torch.load("../trained_models/epoch_20_24889_4.631118524899752.pt")
# )

# # Switch the model to eval mode
# model.eval()

# # Trace the model using the example input
# scripted_model = torch.jit.trace(model, example)

# # Save the scripted model
# scripted_model.save("../trained_models/epoch_20_24889_4.631118524899752_ane.pt")
