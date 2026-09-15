from pathlib import Path

import torch

from dataset import get_batch, TRAIN_TOKENS_PATH
from model.tiny_gpt import TinyGPT
from tokenizer.bpe import load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "tokenizer"
)

TRAIN_TOKENS_FILE = (
    PROJECT_ROOT / TRAIN_TOKENS_PATH
)


def main():

    torch.manual_seed(42)

    # Model configuration
    batch_size = 16
    context_length = 16
    embedding_size = 32
    number_of_heads = 4
    number_of_blocks = 3

    # Training configuration
    learning_rate = 3e-4
    training_steps = 1000
    reporting_interval = 100

    _, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    vocabulary_size = len(vocabulary)

    train_tokens = torch.load(
        TRAIN_TOKENS_FILE,
        map_location="cpu"
    )

    model = TinyGPT(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        context_length=context_length,
        number_of_heads=number_of_heads,
        number_of_blocks=number_of_blocks
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Number of model parameters:")
    print(f"{number_of_parameters:,}")

    print("\nStarting training...")

    model.train()

    running_loss = 0.0
    steps_since_report = 0

    for step in range(1, training_steps + 1):

        input_token_ids, target_token_ids = get_batch(
            token_data=train_tokens,
            batch_size=batch_size,
            context_length=context_length
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(
            input_token_ids=input_token_ids,
            target_token_ids=target_token_ids
        )

        loss.backward()

        optimizer.step()

        # Store only the Python number, not the computation graph.
        running_loss += loss.item()
        steps_since_report += 1

        if (
            step == 1
            or step % reporting_interval == 0
        ):
            average_loss = (
                running_loss / steps_since_report
            )

            print(
                f"Step {step:4d} | "
                f"Average training loss: "
                f"{average_loss:.4f}"
            )

            running_loss = 0.0
            steps_since_report = 0

    print("\nTraining completed.")


if __name__ == "__main__":
    main()