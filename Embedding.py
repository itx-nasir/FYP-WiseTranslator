import torch.nn as nn

'''create Embedding layer with the values vocab_size and d_model
It map each word to a vector
'''


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)
