from pathlib import Path

from tokenizer.bpe import (
    load_tokenizer,
    encode,
    decode,
)


PROJECT_ROOT = Path(__file__).resolve().parent

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
)


TEST_WORDS = [
    "firewall",
    "malware",
    "authentication",
    "credential",
    "cryptography",
    "vulnerability",
]


if __name__ == "__main__":

    merges, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    for word in TEST_WORDS:

        token_ids = encode(
            text=word,
            merges=merges
        )

        reconstructed_word = decode(
            token_ids=token_ids,
            vocabulary=vocabulary
        )

        token_pieces = [
            vocabulary[token_id].decode(
                "utf-8",
                errors="replace"
            )
            for token_id in token_ids
        ]

        byte_count = len(
            word.encode("utf-8")
        )

        token_count = len(token_ids)

        compression_ratio = (
            byte_count / token_count
        )

        print(f"\nWord: {word}")
        print(f"Token IDs: {token_ids}")
        print(f"Token pieces: {token_pieces}")
        print(f"Bytes: {byte_count}")
        print(f"Tokens: {token_count}")
        print(
            f"Bytes per token: "
            f"{compression_ratio:.2f}"
        )
        print(
            f"Decoded correctly: "
            f"{reconstructed_word == word}"
        )