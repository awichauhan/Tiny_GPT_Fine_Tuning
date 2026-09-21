import json
from pathlib import Path
import re
import random

PROJECT_ROOT = Path(__file__).resolve().parent

MITRE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mitre_attack"
    / "enterprise-attack-19.2.json"
)

NIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nist"
    / "glossary-export.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_corpus.txt"
)


USEFUL_MITRE_TYPES = {
    "attack-pattern",
    "malware",
    "tool",
    "intrusion-set",
    "campaign",
    "course-of-action",
}

SECURITY_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "security_corpus.txt"
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

def extract_mitre_entries():
    with MITRE_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    entries = []

    for obj in data["objects"]:

        object_type = obj.get("type")

        if object_type not in USEFUL_MITRE_TYPES:
            continue

        # Ignore revoked/deprecated ATT&CK entries.
        if obj.get("revoked", False):
            continue

        if obj.get("x_mitre_deprecated", False):
            continue

        name = obj.get("name")
        description = obj.get("description")

        if not name or not description:
            continue

        description = clean_text(description)

        entry = (
            f"Name: {name}\n"
            f"Type: {object_type}\n"
            f"Description: {description.strip()}"
        )

        entries.append(entry)

    return entries

def extract_nist_entries():
    with NIST_PATH.open(
        "r",
        encoding="utf-8-sig"
    ) as file:
        data = json.load(file)

    entries = []

    for parent_term in data["parentTerms"]:

        term = parent_term.get("term")
        definitions = parent_term.get(
            "definitions",
            []
        )

        if not term or not definitions:
            continue

        for definition in definitions:

            definition_text = definition.get("text")

            if not definition_text:
                continue

            definition_text = clean_text(
                definition_text
            )

            entry = (
                f"Term: {term}\n"
                f"Definition: {definition_text.strip()}"
            )

            entries.append(entry)

    return entries

def clean_text(text):
    # Convert Markdown links:
    # [Lazarus Group](https://...) -> Lazarus Group
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove MITRE-style citation markers:
    # (Citation: Something)
    text = re.sub(
        r"\(Citation:[^)]+\)",
        "",
        text
    )

    # Collapse repeated spaces/tabs into one space.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Avoid excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()

def remove_duplicates(entries):
    return list(dict.fromkeys(entries))

def build_security_corpus():
    mitre_entries = extract_mitre_entries()
    nist_entries = extract_nist_entries()

    all_entries = (
        mitre_entries
        + nist_entries
    )

    unique_entries = remove_duplicates(
        all_entries
    )

    return unique_entries

def split_entries(
    entries,
    train_fraction=0.9,
    random_seed=42
):
    entries = entries.copy()

    random_generator = random.Random(
        random_seed
    )

    random_generator.shuffle(entries)

    split_index = int(
        len(entries) * train_fraction
    )

    train_entries = entries[:split_index]
    validation_entries = entries[split_index:]

    return train_entries, validation_entries

def split_entries(
    entries,
    train_fraction=0.9,
    random_seed=42
):
    entries = entries.copy()

    random_generator = random.Random(
        random_seed
    )

    random_generator.shuffle(entries)

    split_index = int(
        len(entries) * train_fraction
    )

    train_entries = entries[:split_index]
    validation_entries = entries[split_index:]

    return train_entries, validation_entries

def save_entries(entries, path):
    text = "\n\n".join(entries)

    path.write_text(
        text,
        encoding="utf-8"
    )

if __name__ == "__main__":

    entries = build_security_corpus()

    train_entries, validation_entries = (
        split_entries(entries)
    )

    save_entries(
        entries,
        SECURITY_CORPUS_PATH
    )

    save_entries(
        train_entries,
        SECURITY_TRAIN_PATH
    )

    save_entries(
        validation_entries,
        SECURITY_VALIDATION_PATH
    )

    print("Security corpus prepared.")

    print("\nTotal entries:")
    print(len(entries))

    print("\nTraining entries:")
    print(len(train_entries))

    print("\nValidation entries:")
    print(len(validation_entries))

    print("\nFirst cleaned training entry:\n")
    print(train_entries[0][:1200])