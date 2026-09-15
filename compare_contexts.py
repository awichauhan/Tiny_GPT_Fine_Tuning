import math
from pathlib import Path

import torch

from experiments import get_checkpoint_path
from model.tiny_gpt import TinyGPT
from tokenizer.bpe import (
    load_tokenizer,
    encode,
    decode
)


PROJECT_ROOT = Path(__file__).resolve().parent

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "tokenizer"
)

CONTEXT_EXPERIMENTS = [
    "context_8",
    "context_16",
    "context_32"
]


def load_trained_model(checkpoint_path):

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


def main():

    prompt = "ROMEO:"
    max_new_tokens = 150

    # Keep sampling settings identical.
    temperature = 0.8
    top_k = 20
    random_seed = 42

    merges, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    prompt_token_ids = encode(
        text=prompt,
        merges=merges
    )

    input_token_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long
    )

    print("Context-length comparison")

    print("\nPrompt:")
    print(repr(prompt))

    print("\nSampling settings:")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k}")
    print(f"Random seed: {random_seed}")

    for experiment_name in CONTEXT_EXPERIMENTS:

        checkpoint_path = get_checkpoint_path(
            project_root=PROJECT_ROOT,
            experiment_name=experiment_name
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}"
            )

        model, checkpoint = load_trained_model(
            checkpoint_path
        )

        metrics = checkpoint["metrics"]

        validation_loss = metrics["validation"]

        validation_perplexity = math.exp(
            validation_loss
        )

        # Give every model the same random-number sequence.
        torch.manual_seed(random_seed)

        generated_token_ids = model.generate(
            token_ids=input_token_ids.clone(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=False
        )

        generated_text = decode(
            token_ids=generated_token_ids[0].tolist(),
            vocabulary=vocabulary,
            errors="replace"
        )

        context_length = (
            checkpoint["configuration"]
            ["model"]
            ["context_length"]
        )

        print(
            f"\n--- {experiment_name} ---"
        )

        print(
            f"Context length: {context_length}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation perplexity: "
            f"{validation_perplexity:.2f}"
        )

        print(
            f"Training throughput: "
            f"{metrics['tokens_per_second']:.1f} "
            f"tokens/s"
        )

        print("\nGenerated text:")
        print(generated_text)


if __name__ == "__main__":
    main()