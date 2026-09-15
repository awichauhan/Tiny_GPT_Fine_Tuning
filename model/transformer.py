import torch.nn as nn

from model.transformer_block import TransformerBlock


class TransformerStack(nn.Module):

    def __init__(
        self,
        embedding_size,
        number_of_heads,
        context_length,
        number_of_layers
    ):
        super().__init__()

        if number_of_layers < 1:
            raise ValueError(
                "number_of_layers must be at least 1"
            )

        # Create independent Transformer blocks
        # and execute them in sequence.
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_size=embedding_size,
                    number_of_heads=number_of_heads,
                    context_length=context_length
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(self, x):

        output = self.blocks(x)

        return output