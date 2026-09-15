from pathlib import Path


EXPERIMENTS = {
    "context_8": {
        "batch_size": 32,
        "context_length": 8,
        "embedding_size": 32,
        "number_of_heads": 4,
        "number_of_blocks": 3,
        "learning_rate": 3e-4,
        "target_training_steps": 2000,
        "reporting_interval": 100,
        "evaluation_batches": 20
    },
    "context_16": {
        "batch_size": 16,
        "context_length": 16,
        "embedding_size": 32,
        "number_of_heads": 4,
        "number_of_blocks": 3,
        "learning_rate": 3e-4,
        "target_training_steps": 2000,
        "reporting_interval": 100,
        "evaluation_batches": 20
    },
    "context_32": {
        "batch_size": 8,
        "context_length": 32,
        "embedding_size": 32,
        "number_of_heads": 4,
        "number_of_blocks": 3,
        "learning_rate": 3e-4,
        "target_training_steps": 2000,
        "reporting_interval": 100,
        "evaluation_batches": 20
    }
}


def get_experiment(experiment_name):

    if experiment_name not in EXPERIMENTS:
        available_names = ", ".join(
            EXPERIMENTS.keys()
        )

        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available experiments: {available_names}"
        )

    return EXPERIMENTS[experiment_name].copy()


def get_checkpoint_path(
    project_root,
    experiment_name
):
    return (
        Path(project_root)
        / "artifacts"
        / "checkpoints"
        / f"{experiment_name}.pt"
    )


if __name__ == "__main__":

    for experiment_name, configuration in (
        EXPERIMENTS.items()
    ):
        tokens_per_step = (
            configuration["batch_size"]
            * configuration["context_length"]
        )

        print(f"\nExperiment: {experiment_name}")
        print(
            "Context length:",
            configuration["context_length"]
        )
        print(
            "Batch size:",
            configuration["batch_size"]
        )
        print(
            "Token predictions per step:",
            tokens_per_step
        )