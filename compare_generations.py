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
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
)

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


PROMPTS = [
    "Term: firewall\nDefinition:",
    "Term: malware\nDefinition:",
    "Term: authentication\nDefinition:",
    "Name: Malware\nDescription:",
]


# We keep generation relatively short because
# TinyGPT has a very small model and context window.
MAX_NEW_TOKENS = 60


DECODING_EXPERIMENTS = [
    {
        "name": "Greedy",
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "greedy": True
    },
    {
        "name": "Temperature 0.8",
        "temperature": 0.8,
        "top_k": None,
        "top_p": None,
        "greedy": False
    },
    {
        "name": "Temperature 0.8 + Top-k 20",
        "temperature": 0.8,
        "top_k": 20,
        "top_p": None,
        "greedy": False
    },
    {
        "name": "Temperature 0.8 + Top-p 0.9",
        "temperature": 0.8,
        "top_k": None,
        "top_p": 0.9,
        "greedy": False
    }
]


def load_model(
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model_configuration = (
        checkpoint[
            "configuration"
        ][
            "model"
        ]
    )

    model = TinyGPT(
        **model_configuration
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        model_configuration
    )


def generate_text(
    model,
    prompt,
    merges,
    vocabulary,
    temperature,
    top_k,
    top_p,
    greedy,
    max_new_tokens
):

    # Encode the text prompt into token IDs.
    prompt_token_ids = encode(
        text=prompt,
        merges=merges
    )

    # Show how much of our 16-token context
    # the prompt already consumes.
    print(
        f"Prompt token count: {len(prompt_token_ids)}"
    )

    input_token_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long
    )

    generated_token_ids = model.generate(
        token_ids=input_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy
    )

    generated_text = decode(
        token_ids=generated_token_ids[0].tolist(),
        vocabulary=vocabulary,
        errors="replace"
    )

    return generated_text


def main():

    # -----------------------------------------
    # Load tokenizer
    # -----------------------------------------

    merges, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    # -----------------------------------------
    # Load base model
    # -----------------------------------------

    (
        base_model,
        base_configuration
    ) = load_model(
        BASE_CHECKPOINT_PATH
    )

    # -----------------------------------------
    # Load security fine-tuned model
    # -----------------------------------------

    (
        finetuned_model,
        finetuned_configuration
    ) = load_model(
        FINETUNED_CHECKPOINT_PATH
    )

    # Architectures should be identical.
    #
    # We want to compare parameter values,
    # not different model structures.
    assert (
        base_configuration
        == finetuned_configuration
    )

    print(
        "\n"
        "========== GENERATION COMPARISON =========="
        "\n"
    )

    print(
        "Model configuration:"
    )
    print(
        base_configuration
    )

    print(
        "\nMax new tokens:"
    )
    print(
        MAX_NEW_TOKENS
    )

    # -----------------------------------------
    # Test each prompt
    # -----------------------------------------

    for prompt in PROMPTS:

        print(
            "\n"
            "=========================================="
        )

        print(
            f"PROMPT: {repr(prompt)}"
        )

        print(
            "=========================================="
        )

        # -------------------------------------
        # Test every decoding strategy
        # -------------------------------------

        for experiment in DECODING_EXPERIMENTS:

            print(
                "\n"
                "------------------------------------------"
            )

            print(
                f"DECODING: {experiment['name']}"
            )

            print(
                "------------------------------------------"
            )

            # Reset the seed before BASE generation.
            #
            # This gives us deterministic/reproducible
            # stochastic sampling.
            torch.manual_seed(
                42
            )

            base_output = generate_text(
                model=base_model,
                prompt=prompt,
                merges=merges,
                vocabulary=vocabulary,
                temperature=(
                    experiment[
                        "temperature"
                    ]
                ),
                top_k=(
                    experiment[
                        "top_k"
                    ]
                ),
                top_p=(
                    experiment[
                        "top_p"
                    ]
                ),
                greedy=(
                    experiment[
                        "greedy"
                    ]
                ),
                max_new_tokens=MAX_NEW_TOKENS
            )

            # Reset to the SAME seed before
            # fine-tuned-model generation.
            #
            # The random sampling process therefore
            # begins from the same state.
            #
            # Differences should primarily come from
            # the different model probability
            # distributions.
            torch.manual_seed(
                42
            )

            finetuned_output = generate_text(
                model=finetuned_model,
                prompt=prompt,
                merges=merges,
                vocabulary=vocabulary,
                temperature=(
                    experiment[
                        "temperature"
                    ]
                ),
                top_k=(
                    experiment[
                        "top_k"
                    ]
                ),
                top_p=(
                    experiment[
                        "top_p"
                    ]
                ),
                greedy=(
                    experiment[
                        "greedy"
                    ]
                ),
                max_new_tokens=MAX_NEW_TOKENS
            )

            print(
                "\nBASE MODEL:\n"
            )

            print(
                base_output
            )

            print(
                "\nSECURITY FINE-TUNED MODEL:\n"
            )

            print(
                finetuned_output
            )


if __name__ == "__main__":
    main()