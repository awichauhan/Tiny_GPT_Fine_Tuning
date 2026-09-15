from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_batch, TRAIN_TOKENS_PATH
from tokenizer.bpe import load_tokenizer, decode
from model.transformer_block import TransformerBlock

TOKENIZER_DIRECTORY = Path("artifacts/tokenizer")


class InputEmbedding(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_size,
        context_length
    ):
        super().__init__()

        # Converts token IDs into token embedding vectors.
        self.token_embedding_table = nn.Embedding(
            vocabulary_size,
            embedding_size
        )

        # Converts position IDs into position embedding vectors.
        self.position_embedding_table = nn.Embedding(
            context_length,
            embedding_size
        )

        self.context_length = context_length

    def forward(self, token_ids):

        # token_ids shape: (B, T)
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                "Sequence length exceeds context length"
            )

        # What token is present?
        # Shape: (B, T, C)
        token_embeddings = self.token_embedding_table(
            token_ids
        )

        # Position IDs: [0, 1, 2, ..., T-1]
        # Shape: (T,)
        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        # Where does the token occur?
        # Shape: (T, C)
        position_embeddings = (
            self.position_embedding_table(
                position_ids
            )
        )

        # Add a batch dimension.
        # Shape: (1, T, C)
        position_embeddings = (
            position_embeddings.unsqueeze(0)
        )

        # Use the same position sequence for every batch item.
        # Shape: (B, T, C)
        position_embeddings = position_embeddings.expand(
            batch_size,
            -1,
            -1
        )

        # Combine token identity and token position.
        # Shape: (B, T, C)
        x = token_embeddings + position_embeddings

        # Temporary shape inspection
        print("\nInside input embedding:")
        print("Token IDs:", token_ids.shape)
        print("Token embeddings:", token_embeddings.shape)
        print("Position IDs:", position_ids.shape)
        print("Position embeddings:", position_embeddings.shape)
        print("Combined embeddings:", x.shape)

        return x


if __name__ == "__main__":

    torch.manual_seed(42)

    # Two independent Shakespeare context windows.
    batch_size = 2

    # Sixteen BPE tokens in each sequence.
    context_length = 16

    # Thirty-two embedding features for every token.
    embedding_size = 32

    # Four heads, each with 32 / 4 = 8 features.
    number_of_heads = 4

    # Load the trained BPE vocabulary.
    _, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    vocabulary_size = len(vocabulary)

    # Load all encoded Shakespeare training tokens.
    train_tokens = torch.load(
        TRAIN_TOKENS_PATH,
        map_location="cpu"
    )

    # Select two real Shakespeare context windows.
    input_token_ids, target_token_ids = get_batch(
        token_data=train_tokens,
        batch_size=batch_size,
        context_length=context_length
    )

    print("Vocabulary size:")
    print(vocabulary_size)

    print("\nInput token-ID shape:")
    print(input_token_ids.shape)

    print("\nTarget token-ID shape:")
    print(target_token_ids.shape)

    print("\nFirst sequence token IDs:")
    print(input_token_ids[0])

    # Decode each BPE token individually.
    first_sequence_tokens = [
        decode(
            token_ids=[token_id],
            vocabulary=vocabulary
        )
        for token_id in input_token_ids[0].tolist()
    ]

    print("\nFirst sequence as BPE tokens:")
    print(first_sequence_tokens)

    # Decode the complete context window.
    first_sequence_text = decode(
        token_ids=input_token_ids[0].tolist(),
        vocabulary=vocabulary
    )

    print("\nFirst sequence reconstructed as text:")
    print(repr(first_sequence_text))

    # Create token and position embedding tables.
    input_embedding_layer = InputEmbedding(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        context_length=context_length
    )

    # Convert input token IDs into combined embeddings.
    x = input_embedding_layer(
        input_token_ids
    )

    print("\nToken embedding-table shape:")
    print(
        input_embedding_layer
        .token_embedding_table
        .weight.shape
    )

    print("\nPosition embedding-table shape:")
    print(
        input_embedding_layer
        .position_embedding_table
        .weight.shape
    )

    print("\nCombined input-embedding shape:")
    print(x.shape)

    # Inspect one actual token.
    first_token_id = input_token_ids[0, 0]

    first_token_embedding = (
        input_embedding_layer.token_embedding_table(
            first_token_id
        )
    )

    position_zero = torch.tensor(
        0,
        device=input_token_ids.device
    )

    first_position_embedding = (
        input_embedding_layer.position_embedding_table(
            position_zero
        )
    )

    print("\nFirst token ID:")
    print(first_token_id)

    print("\nFirst token text:")
    print(
        repr(
            decode(
                token_ids=[first_token_id.item()],
                vocabulary=vocabulary
            )
        )
    )

    print("\nFirst token embedding:")
    print(first_token_embedding)

    print("\nPosition-0 embedding:")
    print(first_position_embedding)

    print("\nCombined token + position representation:")
    print(x[0, 0])

    # Verify the addition manually.
    assert torch.allclose(
        x[0, 0],
        first_token_embedding + first_position_embedding
    )

    print("\nToken + position addition check passed.")

    # Pass actual Shakespeare embeddings through
    # one complete Transformer block.
    transformer_block = TransformerBlock(
        embedding_size=embedding_size,
        number_of_heads=number_of_heads,
        context_length=context_length
    )

    transformer_output = transformer_block(x)

    print("\nTransformer-block input shape:")
    print(x.shape)

    print("\nTransformer-block output shape:")
    print(transformer_output.shape)

    assert transformer_output.shape == x.shape

    print("\nTransformer-block shape check passed.")

