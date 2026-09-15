import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_size,
        expansion_factor=4
    ):
        super().__init__()

        hidden_size = (
            embedding_size * expansion_factor
        )

        self.network = nn.Sequential(
            # Expand each token's features.
            nn.Linear(
                embedding_size,
                hidden_size
            ),

            # Add non-linearity.
            nn.GELU(),

            # Return to the original embedding size.
            nn.Linear(
                hidden_size,
                embedding_size
            )
        )

    def forward(self, x):

        output = self.network(x)

        return output