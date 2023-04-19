import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Currently, this model has the following improvements over the simple_model:
# - Teacher forcing: https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
# - Dropout: https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(SimpleTransformer, self).__init__()

        # If Apple Silicon, use MPS. If not, try CUDA. If not, use CPU.
        self.device = "mps" if torch.backends.mps.is_available(
        ) else "cuda" if torch.cuda.is_available() else "cpu"

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
