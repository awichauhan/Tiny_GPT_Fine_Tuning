from pathlib import Path

import torch


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    step,
    configuration,
    metrics
):
    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "configuration": configuration,
        "metrics": metrics,
        "torch_rng_state": torch.get_rng_state()
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

    return checkpoint