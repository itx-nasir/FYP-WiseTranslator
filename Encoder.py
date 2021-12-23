# Import useful libraries
import torch.nn as nn
from FeedForward import FeedForward
from MultiHeadAttention import MultiHeadAttention
from Normalization import Normalization

'''Define EncodingLayer  of Transformer and inherit it from nn.module which
contain the implementation of ecoding layer'''


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        # Lets define encoder with two Normalize layers, 1 attention and one feedforward
        self.normalization_1 = Normalization(d_model)
        self.normalization_2 = Normalization(d_model)
        self.attention = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.feedforward = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.normalization_1(x)
        x = x + self.dropout_1(self.attention(x2, x2, x2, mask))
        x2 = self.normalization_2(x)
        x = x + self.dropout_2(self.feedforward(x2))
        return x
