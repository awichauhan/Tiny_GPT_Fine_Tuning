import argparse
import time
from pathlib import Path

import torch

from checkpoint import (
    load_checkpoint,
    save_checkpoint
)
from dataset import (
    get_batch,
    TRAIN_TOKENS_PATH,
    VALIDATION_TOKENS_PATH
)
from experiments import (
    EXPERIMENTS,
    get_checkpoint_path,
    get_experiment
)
from model.tiny_gpt import TinyGPT
from tokenizer.bpe import load_tokenizer


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


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train a TinyGPT experiment."
    )

    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS.keys()),
        default="context_16",
        help="Experiment configuration to train."
    )

    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model,
    train_tokens,
    validation_tokens,
    batch_size,
    context_length,
    evaluation_batches
):
    losses_by_split = {}

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

    model.train()

    return losses_by_split


def main():

    arguments = parse_arguments()

    experiment_name = arguments.experiment

    experiment = get_experiment(
        experiment_name
    )

    torch.manual_seed(42)

    batch_size = experiment["batch_size"]
    context_length = experiment["context_length"]
    embedding_size = experiment["embedding_size"]
    number_of_heads = experiment["number_of_heads"]
    number_of_blocks = experiment["number_of_blocks"]

    learning_rate = experiment["learning_rate"]

    target_training_steps = (
        experiment["target_training_steps"]
    )

    reporting_interval = (
        experiment["reporting_interval"]
    )

    evaluation_batches = (
        experiment["evaluation_batches"]
    )

    checkpoint_path = get_checkpoint_path(
        project_root=PROJECT_ROOT,
        experiment_name=experiment_name
    )

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

    if checkpoint_path.exists():

        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer
        )

        starting_step = checkpoint["step"]

        print(
            f"Resuming experiment "
            f"'{experiment_name}' from "
            f"step {starting_step}."
        )

    else:
        print(
            f"No checkpoint found for "
            f"'{experiment_name}'. "
            f"Starting fresh training."
        )

    if starting_step >= target_training_steps:
        print(
            f"Experiment '{experiment_name}' "
            f"has already reached "
            f"{target_training_steps} steps."
        )
        return

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nExperiment:")
    print(experiment_name)

    print("\nCheckpoint:")
    print(checkpoint_path)

    print("\nContext length:")
    print(context_length)

    print("\nBatch size:")
    print(batch_size)

    print("\nToken predictions per step:")
    print(batch_size * context_length)

    print("\nNumber of model parameters:")
    print(f"{number_of_parameters:,}")

    print("\nStarting training...")

    model.train()

    training_seconds = 0.0

    for step in range(
        starting_step + 1,
        target_training_steps + 1
    ):

        step_start_time = time.perf_counter()

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

        # This excludes validation time from throughput.
        training_seconds += (
            time.perf_counter() - step_start_time
        )

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

            completed_steps = (
                step - starting_step
            )

            processed_tokens = (
                completed_steps
                * batch_size
                * context_length
            )

            tokens_per_second = (
                processed_tokens / training_seconds
            )

            print(
                f"Step {step:4d} | "
                f"Train: "
                f"{estimated_losses['train']:.4f} | "
                f"Validation: "
                f"{estimated_losses['validation']:.4f} | "
                f"Tokens/s: "
                f"{tokens_per_second:.1f}"
            )

            checkpoint_configuration = {
                "experiment_name": experiment_name,
                "model": model_configuration,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "target_training_steps": (
                    target_training_steps
                )
            }

            checkpoint_metrics = {
                **estimated_losses,
                "tokens_per_second": tokens_per_second
            }

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                configuration=checkpoint_configuration,
                metrics=checkpoint_metrics
            )

            print(
                f"Checkpoint saved: "
                f"{checkpoint_path.name}"
            )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()