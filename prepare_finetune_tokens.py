from pathlib import Path

import torch

from tokenizer.bpe import (
    load_tokenizer,
    encode
)


PROJECT_ROOT = Path(__file__).resolve().parent

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
)

SECURITY_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_train.txt"
)

SECURITY_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_validation.txt"
)

SECURITY_TRAIN_TOKENS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_train_tokens.pt"
)

SECURITY_VALIDATION_TOKENS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_validation_tokens.pt"
)

def encode_file(
    input_path,
    output_path,
    merges
):
    text = input_path.read_text(
        encoding="utf-8"
    )

    token_ids = encode(
        text=text,
        merges=merges
    )

    token_tensor = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    torch.save(
        token_tensor,
        output_path
    )

    return token_tensor

if __name__ == "__main__":

    merges, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    print("Tokenizer vocabulary size:")
    print(len(vocabulary))

    train_tokens = encode_file(
        input_path=SECURITY_TRAIN_PATH,
        output_path=SECURITY_TRAIN_TOKENS_PATH,
        merges=merges
    )

    validation_tokens = encode_file(
        input_path=SECURITY_VALIDATION_PATH,
        output_path=SECURITY_VALIDATION_TOKENS_PATH,
        merges=merges
    )

    print("\nTraining token count:")
    print(len(train_tokens))

    print("\nValidation token count:")
    print(len(validation_tokens))

    print("\nFirst 50 training token IDs:")
    print(train_tokens[:50])