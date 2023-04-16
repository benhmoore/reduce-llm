import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SimpleTransformer(nn.Module):
    """
    A simple transformer model for generative tasks.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the input embeddings and the transformer's hidden size.
        nhead (int): The number of attention heads in the transformer.
        num_layers (int): The number of transformer layers.
        dim_feedforward (int): The dimension of the feedforward network in the transformer.
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super(SimpleTransformer, self).__init__()

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = TransformerEncoder(TransformerEncoderLayer(
            d_model, nhead, dim_feedforward), num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass of the SimpleTransformer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length).

        Returns:
            x (torch.Tensor): The output tensor of shape (batch_size, seq_length, vocab_size).
        """

        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

    def generate(self, input_ids, max_length, do_sample=True, temperature=1.0):
        with torch.no_grad():
            generated_text = input_ids
            for _ in range(max_length - input_ids.shape[-1]):
                # Assuming forward() returns logits
                logits = self.forward(generated_text)
                if do_sample:
                    logits = logits[:, -1] / temperature
                    next_token = torch.multinomial(
                        F.softmax(logits, dim=-1), num_samples=1)
                else:
                    next_token = torch.argmax(
                        logits[:, -1], dim=-1).unsqueeze(-1)
                generated_text = torch.cat(
                    (generated_text, next_token), dim=-1)
            return generated_text
