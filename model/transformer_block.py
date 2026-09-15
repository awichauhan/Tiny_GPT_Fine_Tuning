import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size,
        number_of_heads,
        context_length
    ):
        super().__init__()

        # Normalization before attention.
        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.attention = MultiHeadAttention(
            embedding_size=embedding_size,
            number_of_heads=number_of_heads,
            context_length=context_length
        )

        # Normalization before the feed-forward network.
        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = FeedForward(
            embedding_size=embedding_size
        )

    def forward(self, x):

        # Normalize → attention → residual addition
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Normalize → FFN → residual addition
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x