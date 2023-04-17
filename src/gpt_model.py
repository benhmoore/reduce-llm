import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        gpt_config = GPT2Config(vocab_size=vocab_size,
                                n_embd=d_model,
                                n_head=nhead,
                                n_layer=num_layers,
                                n_positions=dim_feedforward,
                                n_ctx=dim_feedforward,
                                use_cache=False,
                                dropout=dropout)
        self.gpt = GPT2LMHeadModel(config=gpt_config)

    def forward(self, x):
        x = self.gpt(input_ids=x, return_dict=True).logits
        return x

    def generate(self, input_ids, max_length, do_sample=True, temperature=1.0):
        with torch.no_grad():
            generated_text = self.gpt.generate(input_ids,
                                               max_length=max_length,
                                               do_sample=do_sample,
                                               temperature=temperature)
        return generated_text
