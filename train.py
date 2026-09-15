from pathlib import Path

import torch

from dataset import (
    get_batch,
    TRAIN_TOKENS_PATH,
    VALIDATION_TOKENS_PATH
)
from model.tiny_gpt import TinyGPT
from tokenizer.bpe import load_tokenizer

from checkpoint import (
    save_checkpoint,
    load_checkpoint
)



PROJECT_ROOT = Path(__file__).resolve().parent

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "tokenizer"
)

TRAIN_TOKENS_FILE = (
    PROJECT_ROOT / TRAIN_TOKENS_PATH
)

VALIDATION_TOKENS_FILE = (
    PROJECT_ROOT / VALIDATION_TOKENS_PATH
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "tiny_gpt.pt"
)


@torch.no_grad()
def estimate_loss(
    model,
    train_tokens,
    validation_tokens,
    batch_size,
    context_length,
    evaluation_batches
):
    """
    Estimate average loss on multiple training
    and validation batches.
    """

    losses_by_split = {}

    # Disable training-specific behaviour.
    model.eval()

    datasets = {
        "train": train_tokens,
        "validation": validation_tokens
    }

    for split_name, token_data in datasets.items():

        batch_losses = torch.zeros(
            evaluation_batches
        )

        for batch_index in range(evaluation_batches):

            input_token_ids, target_token_ids = get_batch(
                token_data=token_data,
                batch_size=batch_size,
                context_length=context_length
            )

            _, loss = model(
                input_token_ids=input_token_ids,
                target_token_ids=target_token_ids
            )

            batch_losses[batch_index] = loss.item()

        losses_by_split[split_name] = (
            batch_losses.mean().item()
        )

    # Return the model to training mode.
    model.train()

    return losses_by_split


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
    evaluation_batches = 20

    _, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    vocabulary_size = len(vocabulary)

    train_tokens = torch.load(
        TRAIN_TOKENS_FILE,
        map_location="cpu"
    )

    validation_tokens = torch.load(
        VALIDATION_TOKENS_FILE,
        map_location="cpu"
    )

    model_configuration = {
        "vocabulary_size": vocabulary_size,
        "embedding_size": embedding_size,
        "context_length": context_length,
        "number_of_heads": number_of_heads,
        "number_of_blocks": number_of_blocks
    }

    model = TinyGPT(
        **model_configuration
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    starting_step = 0

    if CHECKPOINT_PATH.exists():

        checkpoint = load_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer
        )

        starting_step = checkpoint["step"]

        print(
            f"Resuming training from step "
            f"{starting_step}."
        )

    else:
        print(
            "No checkpoint found. "
            "Starting fresh training."
        )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Number of model parameters:")
    print(f"{number_of_parameters:,}")

    print("\nStarting training...")

    model.train()

    ending_step = starting_step + training_steps

    for step in range(
            starting_step + 1,
            ending_step + 1
    ):

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

        if (
            step == 1
            or step % reporting_interval == 0
        ):
            estimated_losses = estimate_loss(
                model=model,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                batch_size=batch_size,
                context_length=context_length,
                evaluation_batches=evaluation_batches
            )

            print(
                f"Step {step:4d} | "
                f"Train loss: "
                f"{estimated_losses['train']:.4f} | "
                f"Validation loss: "
                f"{estimated_losses['validation']:.4f}"
            )

            configuration = {
                "model": model_configuration,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }

            save_checkpoint(
                checkpoint_path=CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                step=step,
                configuration=configuration,
                metrics=estimated_losses
            )

            print(
                f"Checkpoint saved at step {step}."
            )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()