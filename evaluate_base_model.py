from pathlib import Path

import torch

from dataset import get_batch
from model.tiny_gpt import TinyGPT


PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "tiny_gpt.pt"
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


@torch.no_grad()
def estimate_loss(
    model,
    token_data,
    batch_size,
    context_length,
    evaluation_batches=100
):
    losses = []

    model.eval()

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

        losses.append(loss.item())

    return sum(losses) / len(losses)


if __name__ == "__main__":

    torch.manual_seed(42)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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

    shakespeare_validation_tokens = torch.load(
        SHAKESPEARE_VALIDATION_PATH,
        map_location="cpu"
    )

    security_validation_tokens = torch.load(
        SECURITY_VALIDATION_PATH,
        map_location="cpu"
    )

    batch_size = 32

    context_length = (
        model_configuration["context_length"]
    )

    shakespeare_loss = estimate_loss(
        model=model,
        token_data=shakespeare_validation_tokens,
        batch_size=batch_size,
        context_length=context_length
    )

    security_loss = estimate_loss(
        model=model,
        token_data=security_validation_tokens,
        batch_size=batch_size,
        context_length=context_length
    )

    print("Base checkpoint step:")
    print(checkpoint["step"])

    print("\nModel configuration:")
    print(model_configuration)

    print("\nBase model validation losses:")

    print(
        f"Shakespeare validation loss: "
        f"{shakespeare_loss:.4f}"
    )

    print(
        f"Security validation loss: "
        f"{security_loss:.4f}"
    )