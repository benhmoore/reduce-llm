import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Currently, this model has the following improvements over the improved_simple_model:
# - ReduceLROnPlateau: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
# ReduceLROnPlateau is a scheduler that reduces the learning rate when a metric has stopped improving.


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout=dropout
            ),
            num_layers
        )

        self.fc = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and target is not None:
            # Apply teacher forcing
            use_teacher_forcing = (
                torch.rand(1).item() < teacher_forcing_ratio
            )
            if use_teacher_forcing:
                x = torch.cat([x[:, :-1], target[:, 1:]], dim=1)

        return x

    def generate(self, input_ids, max_length, do_sample=True, temperature=1.0):

        # torch.no_grad() is just a context manager that disables gradient calculation
        with torch.no_grad():
            generated_text = input_ids
            for _ in range(max_length - input_ids.shape[-1]):
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
