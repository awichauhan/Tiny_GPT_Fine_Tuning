from pathlib import Path

import torch

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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "tiny_gpt.pt"
)


def main():

    prompt = "ROMEO:"
    max_new_tokens = 150

    experiments = [
        {
            "name": "Greedy",
            "temperature": 1.0,
            "top_k": None,
            "greedy": True
        },
        {
            "name": "Temperature 0.5",
            "temperature": 0.5,
            "top_k": None,
            "greedy": False
        },
        {
            "name": "Temperature 0.8",
            "temperature": 0.8,
            "top_k": None,
            "greedy": False
        },
        {
            "name": "Temperature 1.2",
            "temperature": 1.2,
            "top_k": None,
            "greedy": False
        },
        {
            "name": "Temperature 0.8 + Top-k 20",
            "temperature": 0.8,
            "top_k": 20,
            "greedy": False
        }
    ]

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # This creates the missing `merges` variable.
    merges, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

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

    model.eval()

    prompt_token_ids = encode(
        text=prompt,
        merges=merges
    )

    input_token_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long
    )

    print("Loaded checkpoint step:")
    print(checkpoint["step"])

    print("\nPrompt:")
    print(repr(prompt))

    for experiment in experiments:

        # Same seed makes comparisons reproducible.
        torch.manual_seed(42)

        generated_token_ids = model.generate(
            token_ids=input_token_ids.clone(),
            max_new_tokens=max_new_tokens,
            temperature=experiment["temperature"],
            top_k=experiment["top_k"],
            greedy=experiment["greedy"]
        )

        generated_text = decode(
            token_ids=generated_token_ids[0].tolist(),
            vocabulary=vocabulary,
            errors="replace"
        )

        print(f"\n--- {experiment['name']} ---")
        print(generated_text)


if __name__ == "__main__":
    main()