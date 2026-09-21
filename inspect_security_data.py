import json
from pathlib import Path
from collections import Counter


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


def inspect_mitre():
    print("\n========== MITRE ATT&CK ==========\n")

    with MITRE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print("Top-level type:")
    print(type(data))

    print("\nTop-level keys:")
    print(data.keys())

    objects = data["objects"]

    print("\nNumber of objects:")
    print(len(objects))

    object_types = Counter(
        obj.get("type")
        for obj in objects
    )

    print("\nObject types:")
    for object_type, count in object_types.most_common():
        print(object_type, count)

    # Find an attack-pattern object because these are
    # ATT&CK techniques and contain useful security text.
    attack_pattern = next(
        obj
        for obj in objects
        if obj.get("type") == "attack-pattern"
    )

    print("\nExample attack-pattern keys:")
    print(attack_pattern.keys())

    print("\nExample technique name:")
    print(attack_pattern.get("name"))

    print("\nExample description:")
    print(
        attack_pattern.get("description", "")[:1000]
    )


def inspect_nist():
    print("\n========== NIST GLOSSARY ==========\n")

    with NIST_PATH.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    print("Top-level type:")
    print(type(data))

    if isinstance(data, dict):
        print("\nTop-level keys:")
        print(data.keys())

        # Show the type of every top-level value.
        for key, value in data.items():
            print(
                f"{key}: {type(value).__name__}"
            )

    elif isinstance(data, list):
        print("\nNumber of entries:")
        print(len(data))

        if data:
            print("\nFirst entry:")
            print(data[0])

    parent_terms = data["parentTerms"]

    print("\nNumber of parent terms:")
    print(len(parent_terms))

    first_term = parent_terms[0]

    print("\nFirst parent term type:")
    print(type(first_term))

    print("\nFirst parent term keys:")
    print(first_term.keys())

    print("\nFirst parent term:")
    print(first_term)


if __name__ == "__main__":
    inspect_mitre()
    inspect_nist()