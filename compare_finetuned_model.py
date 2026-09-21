from pathlib import Path

import torch

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

SHAKESPEARE_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_tokens.pt"
)

SECURITY_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_validation_tokens.pt"
)


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model_configuration = (
        checkpoint["configuration"]["model"]
    )

    model = TinyGPT(
        **model_configuration
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


@torch.no_grad()
def estimate_loss(
    model,
    token_data,
    batch_size,
    context_length,
    evaluation_batches,
    random_seed
):
    # Reset seed so different models receive
    # exactly the same sampled validation windows.
    torch.manual_seed(random_seed)

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

    return sum(losses) / len(losses)


if __name__ == "__main__":

    batch_size = 32
    evaluation_batches = 100
    random_seed = 42

    # -----------------------------
    # Load checkpoints
    # -----------------------------

    base_model, base_checkpoint = load_model(
        BASE_CHECKPOINT_PATH
    )

    finetuned_model, finetuned_checkpoint = load_model(
        FINETUNED_CHECKPOINT_PATH
    )

    base_configuration = (
        base_checkpoint["configuration"]["model"]
    )

    finetuned_configuration = (
        finetuned_checkpoint["configuration"]["model"]
    )

    # Architecture should be unchanged.
    assert (
        base_configuration
        == finetuned_configuration
    )

    context_length = (
        base_configuration["context_length"]
    )

    # -----------------------------
    # Load validation datasets
    # -----------------------------

    shakespeare_validation_tokens = torch.load(
        SHAKESPEARE_VALIDATION_PATH,
        map_location="cpu"
    )

    security_validation_tokens = torch.load(
        SECURITY_VALIDATION_PATH,
        map_location="cpu"
    )

    # -----------------------------
    # Base model
    # -----------------------------

    base_security_loss = estimate_loss(
        model=base_model,
        token_data=security_validation_tokens,
        batch_size=batch_size,
        context_length=context_length,
        evaluation_batches=evaluation_batches,
        random_seed=random_seed
    )

    base_shakespeare_loss = estimate_loss(
        model=base_model,
        token_data=shakespeare_validation_tokens,
        batch_size=batch_size,
        context_length=context_length,
        evaluation_batches=evaluation_batches,
        random_seed=random_seed
    )

    # -----------------------------
    # Fine-tuned model
    # -----------------------------

    finetuned_security_loss = estimate_loss(
        model=finetuned_model,
        token_data=security_validation_tokens,
        batch_size=batch_size,
        context_length=context_length,
        evaluation_batches=evaluation_batches,
        random_seed=random_seed
    )

    finetuned_shakespeare_loss = estimate_loss(
        model=finetuned_model,
        token_data=shakespeare_validation_tokens,
        batch_size=batch_size,
        context_length=context_length,
        evaluation_batches=evaluation_batches,
        random_seed=random_seed
    )

    # -----------------------------
    # Differences
    # -----------------------------

    security_change = (
        finetuned_security_loss
        - base_security_loss
    )

    shakespeare_change = (
        finetuned_shakespeare_loss
        - base_shakespeare_loss
    )

    security_improvement_percent = (
        (
            base_security_loss
            - finetuned_security_loss
        )
        / base_security_loss
        * 100
    )

    shakespeare_degradation_percent = (
        (
            finetuned_shakespeare_loss
            - base_shakespeare_loss
        )
        / base_shakespeare_loss
        * 100
    )

    # -----------------------------
    # Results
    # -----------------------------

    print("\n========== CONTROLLED EVALUATION ==========\n")

    print(f"Evaluation batches: {evaluation_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Context length: {context_length}")
    print(f"Random seed: {random_seed}")

    print("\nBase model:")
    print(
        f"Security validation loss: "
        f"{base_security_loss:.4f}"
    )
    print(
        f"Shakespeare validation loss: "
        f"{base_shakespeare_loss:.4f}"
    )

    print("\nSecurity fine-tuned model:")
    print(
        f"Security validation loss: "
        f"{finetuned_security_loss:.4f}"
    )
    print(
        f"Shakespeare validation loss: "
        f"{finetuned_shakespeare_loss:.4f}"
    )

    print("\nChanges after fine-tuning:")

    print(
        f"Security loss change: "
        f"{security_change:+.4f}"
    )

    print(
        f"Security loss improvement: "
        f"{security_improvement_percent:.2f}%"
    )

    print(
        f"Shakespeare loss change: "
        f"{shakespeare_change:+.4f}"
    )

    print(
        f"Shakespeare loss degradation: "
        f"{shakespeare_degradation_percent:.2f}%"
    )