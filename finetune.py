from pathlib import Path

import torch

from checkpoint import save_checkpoint
from dataset import get_batch
from model.tiny_gpt import TinyGPT


PROJECT_ROOT = Path(__file__).resolve().parent

BASE_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "tiny_gpt.pt"
)

FINETUNED_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "security_finetuned.pt"
)

SECURITY_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_train_tokens.pt"
)

SECURITY_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_validation_tokens.pt"
)

SHAKESPEARE_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_tokens.pt"
)


@torch.no_grad()
def estimate_loss(
    model,
    token_data,
    batch_size,
    context_length,
    evaluation_batches=20
):
    model.eval()

    losses = []

    for _ in range(evaluation_batches):

        input_batch, target_batch = get_batch(
            token_data=token_data,
            batch_size=batch_size,
            context_length=context_length
        )

        _, loss = model(
            input_token_ids=input_batch,
            target_token_ids=target_batch
        )

        losses.append(
            loss.item()
        )

    model.train()

    return sum(losses) / len(losses)


def main():

    torch.manual_seed(42)

    # -----------------------------
    # Fine-tuning configuration
    # -----------------------------

    batch_size = 16
    learning_rate = 1e-4
    training_steps = 500
    reporting_interval = 100
    evaluation_batches = 20

    # -----------------------------
    # Load pretrained checkpoint
    # -----------------------------

    base_checkpoint = torch.load(
        BASE_CHECKPOINT_PATH,
        map_location="cpu"
    )

    model_configuration = (
        base_checkpoint["configuration"]["model"]
    )

    context_length = (
        model_configuration["context_length"]
    )

    model = TinyGPT(
        **model_configuration
    )

    model.load_state_dict(
        base_checkpoint["model_state_dict"]
    )

    print("Loaded pretrained checkpoint:")
    print(BASE_CHECKPOINT_PATH.name)

    print("\nPretraining checkpoint step:")
    print(base_checkpoint["step"])

    print("\nModel configuration:")
    print(model_configuration)

    # -----------------------------
    # Load fine-tuning data
    # -----------------------------

    security_train_tokens = torch.load(
        SECURITY_TRAIN_PATH,
        map_location="cpu"
    )

    security_validation_tokens = torch.load(
        SECURITY_VALIDATION_PATH,
        map_location="cpu"
    )

    shakespeare_validation_tokens = torch.load(
        SHAKESPEARE_VALIDATION_PATH,
        map_location="cpu"
    )

    # -----------------------------
    # NEW optimizer
    # -----------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nFine-tuning configuration:")

    print("Learning rate:")
    print(learning_rate)

    print("\nBatch size:")
    print(batch_size)

    print("\nContext length:")
    print(context_length)

    print("\nTraining steps:")
    print(training_steps)

    print("\nTrainable parameters:")
    print(f"{number_of_parameters:,}")

    # -----------------------------
    # Baseline before first update
    # -----------------------------

    security_loss_before = estimate_loss(
        model=model,
        token_data=security_validation_tokens,
        batch_size=batch_size,
        context_length=context_length,
        evaluation_batches=evaluation_batches
    )

    shakespeare_loss_before = estimate_loss(
        model=model,
        token_data=shakespeare_validation_tokens,
        batch_size=batch_size,
        context_length=context_length,
        evaluation_batches=evaluation_batches
    )

    print("\nBefore fine-tuning:")

    print(
        f"Security validation loss: "
        f"{security_loss_before:.4f}"
    )

    print(
        f"Shakespeare validation loss: "
        f"{shakespeare_loss_before:.4f}"
    )

    # -----------------------------
    # Fine-tuning loop
    # -----------------------------

    model.train()

    for step in range(
        1,
        training_steps + 1
    ):

        input_batch, target_batch = get_batch(
            token_data=security_train_tokens,
            batch_size=batch_size,
            context_length=context_length
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        _, training_loss = model(
            input_token_ids=input_batch,
            target_token_ids=target_batch
        )

        training_loss.backward()

        optimizer.step()

        # -------------------------
        # Periodic evaluation
        # -------------------------

        if (
            step == 1
            or step % reporting_interval == 0
        ):

            security_validation_loss = estimate_loss(
                model=model,
                token_data=security_validation_tokens,
                batch_size=batch_size,
                context_length=context_length,
                evaluation_batches=evaluation_batches
            )

            shakespeare_validation_loss = estimate_loss(
                model=model,
                token_data=shakespeare_validation_tokens,
                batch_size=batch_size,
                context_length=context_length,
                evaluation_batches=evaluation_batches
            )

            print(
                f"\nStep {step:4d}"
                f" | Train: {training_loss.item():.4f}"
                f" | Security val: "
                f"{security_validation_loss:.4f}"
                f" | Shakespeare val: "
                f"{shakespeare_validation_loss:.4f}"
            )

    # -----------------------------
    # Save fine-tuned checkpoint
    # -----------------------------

    checkpoint_configuration = {
        "base_checkpoint": BASE_CHECKPOINT_PATH.name,
        "model": model_configuration,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_steps": training_steps
    }

    checkpoint_metrics = {
        "security_validation_loss": (
            security_validation_loss
        ),
        "shakespeare_validation_loss": (
            shakespeare_validation_loss
        )
    }

    save_checkpoint(
        checkpoint_path=FINETUNED_CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        step=training_steps,
        configuration=checkpoint_configuration,
        metrics=checkpoint_metrics
    )

    print("\nFine-tuning completed.")

    print("\nFine-tuned checkpoint saved to:")
    print(FINETUNED_CHECKPOINT_PATH)


if __name__ == "__main__":
    main()