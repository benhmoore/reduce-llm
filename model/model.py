import torch
import torch.nn as nn
from transformers import Transformer
import torch.optim as optim


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = Transformer(
            d_model, nhead, num_layers, dim_feedforward)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


# Hyperparameters
vocab_size = tokenizer.vocab_size
d_model = 256
nhead = 8
num_layers = 3
dim_feedforward = 512

model = SimpleTransformer(vocab_size, d_model, nhead,
                          num_layers, dim_feedforward)
