from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_batch, TRAIN_TOKENS_PATH
from model.embeddings import InputEmbedding
from model.transformer import TransformerStack
from tokenizer.bpe import load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
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

        # Convert each token's hidden representation
        # into one score for every token in the vocabulary.
        #
        # (B, T, C) -> (B, T, V)
        self.vocabulary_projection = nn.Linear(
            embedding_size,
            vocabulary_size
        )

        self.context_length = context_length

    def forward(
        self,
        input_token_ids,
        target_token_ids=None
    ):
        # Token IDs:
        # (B, T)
        #
        # Embeddings:
        # (B, T, C)
        x = self.input_embedding(
            input_token_ids
        )

        # Transformer blocks preserve:
        # (B, T, C)
        x = self.transformer_stack(
            x
        )

        # Final normalization:
        # (B, T, C)
        x = self.final_layer_norm(
            x
        )

        # Vocabulary projection:
        # (B, T, C) -> (B, T, V)
        logits = self.vocabulary_projection(
            x
        )

        loss = None

        # During training/evaluation we provide targets
        # and calculate next-token cross-entropy loss.
        if target_token_ids is not None:

            (
                batch_size,
                sequence_length,
                vocabulary_size
            ) = logits.shape

            # Treat every token position as one
            # classification example.
            #
            # (B, T, V)
            #       ↓
            # (B*T, V)
            flattened_logits = logits.reshape(
                batch_size * sequence_length,
                vocabulary_size
            )

            # (B, T)
            #   ↓
            # (B*T)
            flattened_targets = (
                target_token_ids.reshape(
                    batch_size * sequence_length
                )
            )

            loss = F.cross_entropy(
                flattened_logits,
                flattened_targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        greedy=False
    ):
        """
        Autoregressively generate tokens.

        Supported decoding methods:

        greedy=True
            Always choose the highest-scoring token.

        temperature
            Controls how sharp/flat the probability
            distribution becomes before sampling.

        top_k
            Restrict sampling to the K highest-scoring
            candidate tokens.

        top_p
            Restrict sampling to the smallest group of
            tokens whose cumulative probability reaches
            probability mass P.

        max_new_tokens
            Maximum number of new tokens to generate.
        """

        # Temperature only matters for sampling.
        if not greedy and temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero"
            )

        if top_k is not None and top_k < 1:
            raise ValueError(
                "top_k must be at least one"
            )

        if top_p is not None:
            if not 0 < top_p <= 1:
                raise ValueError(
                    "top_p must be greater than 0 "
                    "and at most 1"
                )

        # Generate one token at a time.
        for _ in range(max_new_tokens):

            # The model can only attend to the latest
            # context_length tokens.
            #
            # This acts as a rolling/sliding context window.
            current_context = token_ids[
                :,
                -self.context_length:
            ]

            # Forward pass.
            logits, _ = self(
                current_context
            )

            # logits:
            # (B, T, V)
            #
            # For generation we only need the prediction
            # made at the final sequence position.
            #
            # Result:
            # (B, V)
            next_token_logits = logits[
                :,
                -1,
                :
            ]

            # -----------------------------------------
            # GREEDY DECODING
            # -----------------------------------------

            if greedy:

                next_token_id = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True
                )

            # -----------------------------------------
            # SAMPLING
            # -----------------------------------------

            else:

                # Temperature scaling.
                #
                # Lower temperature:
                # sharper probability distribution.
                #
                # Higher temperature:
                # flatter probability distribution.
                next_token_logits = (
                    next_token_logits
                    / temperature
                )

                # -------------------------------------
                # TOP-K FILTERING
                # -------------------------------------

                if top_k is not None:

                    number_to_keep = min(
                        top_k,
                        next_token_logits.shape[-1]
                    )

                    top_values, _ = torch.topk(
                        next_token_logits,
                        k=number_to_keep,
                        dim=-1
                    )

                    # Smallest logit among the top-k.
                    cutoff = (
                        top_values[
                            :,
                            -1
                        ]
                        .unsqueeze(-1)
                    )

                    # Anything below the cutoff becomes
                    # impossible after softmax.
                    next_token_logits = (
                        next_token_logits.masked_fill(
                            next_token_logits < cutoff,
                            float("-inf")
                        )
                    )

                # Convert logits into probabilities.
                probabilities = torch.softmax(
                    next_token_logits,
                    dim=-1
                )

                # -------------------------------------
                # TOP-P / NUCLEUS FILTERING
                # -------------------------------------

                if top_p is not None:

                    # Sort tokens from most probable
                    # to least probable.
                    (
                        sorted_probabilities,
                        sorted_indices
                    ) = torch.sort(
                        probabilities,
                        descending=True,
                        dim=-1
                    )

                    # Prefix/cumulative probability mass.
                    cumulative_probabilities = (
                        torch.cumsum(
                            sorted_probabilities,
                            dim=-1
                        )
                    )

                    # Initially mark tokens appearing
                    # after cumulative probability
                    # exceeds top_p.
                    tokens_to_remove = (
                        cumulative_probabilities
                        > top_p
                    )

                    # Shift the mask right.
                    #
                    # This keeps the first token that
                    # pushes cumulative probability
                    # over the threshold.
                    tokens_to_remove[
                        :,
                        1:
                    ] = (
                        tokens_to_remove[
                            :,
                            :-1
                        ].clone()
                    )

                    # Always retain at least the
                    # highest-probability token.
                    tokens_to_remove[
                        :,
                        0
                    ] = False

                    # Removed candidates get probability 0.
                    sorted_probabilities = (
                        sorted_probabilities.masked_fill(
                            tokens_to_remove,
                            0.0
                        )
                    )

                    # Since probability mass was removed,
                    # normalize remaining probabilities
                    # back to a sum of 1.
                    sorted_probabilities = (
                        sorted_probabilities
                        / sorted_probabilities.sum(
                            dim=-1,
                            keepdim=True
                        )
                    )

                    # Sample an index in the SORTED
                    # probability array.
                    sampled_position = (
                        torch.multinomial(
                            sorted_probabilities,
                            num_samples=1
                        )
                    )

                    # Convert sorted position back to
                    # the actual vocabulary token ID.
                    next_token_id = torch.gather(
                        sorted_indices,
                        dim=-1,
                        index=sampled_position
                    )

                else:

                    # Standard sampling from the full
                    # probability distribution,
                    # or from top-k filtered probabilities.
                    next_token_id = torch.multinomial(
                        probabilities,
                        num_samples=1
                    )

            # Append predicted token to the sequence.
            #
            # Existing:
            # [token1 token2 token3]
            #
            # New:
            # [token1 token2 token3 token4]
            token_ids = torch.cat(
                [
                    token_ids,
                    next_token_id
                ],
                dim=1
            )

        return token_ids


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

    vocabulary_size = len(
        vocabulary
    )

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
    print(
        input_token_ids.shape
    )

    print("\nTarget shape:")
    print(
        target_token_ids.shape
    )

    print("\nLogits shape:")
    print(
        logits.shape
    )

    print(
        "\nVocabulary-projection weight shape:"
    )
    print(
        model.vocabulary_projection.weight.shape
    )

    predicted_token_id = (
        logits[
            0,
            -1
        ]
        .argmax()
        .item()
    )

    print(
        "\nPredicted token ID at final position:"
    )
    print(
        predicted_token_id
    )

    predicted_token_bytes = (
        vocabulary[
            predicted_token_id
        ]
    )

    print(
        "\nPredicted token bytes:"
    )
    print(
        repr(
            predicted_token_bytes
        )
    )

    print(
        "\nSafely displayed token:"
    )
    print(
        repr(
            predicted_token_bytes.decode(
                "utf-8",
                errors="replace"
            )
        )
    )

    assert logits.shape == (
        batch_size,
        context_length,
        vocabulary_size
    )

    print(
        "\nTinyGPT forward-pass checks passed."
    )

    print(
        "\nCross-entropy loss:"
    )
    print(
        loss.item()
    )

    assert loss.ndim == 0
    assert torch.isfinite(
        loss
    )

    print(
        "\nLoss calculation checks passed."
    )