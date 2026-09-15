from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_batch, TRAIN_TOKENS_PATH
from model.embeddings import InputEmbedding
from model.transformer import TransformerStack
from tokenizer.bpe import load_tokenizer, decode
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "tokenizer"
)


class TinyGPT(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_size,
        context_length,
        number_of_heads,
        number_of_blocks
    ):
        super().__init__()

        self.input_embedding = InputEmbedding(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            context_length=context_length
        )

        self.transformer_stack = TransformerStack(
            embedding_size=embedding_size,
            number_of_heads=number_of_heads,
            context_length=context_length,
            number_of_layers=number_of_blocks
        )

        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # Convert C features into one score for every vocabulary token.
        self.vocabulary_projection = nn.Linear(
            embedding_size,
            vocabulary_size
        )

    def forward(
            self,
            input_token_ids,
            target_token_ids=None
    ):
        # (B, T) → (B, T, C)
        x = self.input_embedding(input_token_ids)

        # (B, T, C) → (B, T, C)
        x = self.transformer_stack(x)

        # (B, T, C) → (B, T, C)
        x = self.final_layer_norm(x)

        # (B, T, C) → (B, T, V)
        logits = self.vocabulary_projection(x)

        loss = None

        if target_token_ids is not None:
            batch_size, sequence_length, vocabulary_size = (
                logits.shape
            )

            # Treat every token position as one classification example.
            # (B, T, V) → (B*T, V)
            flattened_logits = logits.reshape(
                batch_size * sequence_length,
                vocabulary_size
            )

            # (B, T) → (B*T)
            flattened_targets = target_token_ids.reshape(
                batch_size * sequence_length
            )

            loss = F.cross_entropy(
                flattened_logits,
                flattened_targets
            )

        return logits, loss


if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 2
    context_length = 16
    embedding_size = 32
    number_of_heads = 4
    number_of_blocks = 3

    merges, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    vocabulary_size = len(vocabulary)

    train_tokens = torch.load(
        TRAIN_TOKENS_PATH,
        map_location="cpu"
    )

    input_token_ids, target_token_ids = get_batch(
        token_data=train_tokens,
        batch_size=batch_size,
        context_length=context_length
    )

    model = TinyGPT(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        context_length=context_length,
        number_of_heads=number_of_heads,
        number_of_blocks=number_of_blocks
    )

    logits, loss = model(
        input_token_ids=input_token_ids,
        target_token_ids=target_token_ids
    )

    print("\nInput shape:")
    print(input_token_ids.shape)

    print("\nTarget shape:")
    print(target_token_ids.shape)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nVocabulary-projection weight shape:")
    print(model.vocabulary_projection.weight.shape)

    predicted_token_id = logits[0, -1].argmax().item()

    print("\nPredicted token ID at final position:")
    print(predicted_token_id)

    predicted_token_bytes = vocabulary[predicted_token_id]

    print("\nPredicted token bytes:")
    print(repr(predicted_token_bytes))

    print("\nSafely displayed token:")
    print(repr(
        predicted_token_bytes.decode(
            "utf-8",
            errors="replace"
        )
    ))

    assert logits.shape == (
        batch_size,
        context_length,
        vocabulary_size
    )

    print("\nTinyGPT forward-pass checks passed.")

    print("\nCross-entropy loss:")
    print(loss.item())

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("\nLoss calculation checks passed.")