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

    torch.manual_seed(42)

    prompt = "ROMEO:"
    max_new_tokens = 200
    temperature = 0.8

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

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

    generated_token_ids = model.generate(
        token_ids=input_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    generated_text = decode(
        token_ids=generated_token_ids[0].tolist(),
        vocabulary=vocabulary,
        errors="replace"
    )

    print("Loaded checkpoint step:")
    print(checkpoint["step"])

    print("\nPrompt:")
    print(repr(prompt))

    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main()