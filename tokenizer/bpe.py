import json
from pathlib import Path


def get_pair_counts(token_ids):
    pair_counts = {}
    for pair in zip(token_ids, token_ids[1:]):
        pair_counts[pair] = pair_counts.get(pair, 0)+ 1

    return pair_counts

def merge_pair(token_ids, pair_to_merge, new_token_id):
    merged_token_ids = []
    index = 0
    while index < len(token_ids):
        if (
            index < len(token_ids) -1
            and token_ids[index] == pair_to_merge[0]
            and token_ids[index+1] == pair_to_merge[1]
        ):
            merged_token_ids.append(new_token_id)
            index+=2
        else:
            merged_token_ids.append(token_ids[index])
            index+=1
    return merged_token_ids

def train_bpe(text, vocabulary_size):
    if vocabulary_size < 256:
        raise ValueError(
            "Vocabulary size must be at least 256"
        )
    token_ids = list(text.encode("utf-8"))
    merges = {}

    for new_token_id in range(256, vocabulary_size):
        pair_counts = get_pair_counts(token_ids)
        if not pair_counts:
            break

        most_frequent_pair = max(
            pair_counts,
            key= pair_counts.get
        )
        frequency = pair_counts[most_frequent_pair]

        token_ids = merge_pair(
            token_ids= token_ids,
            pair_to_merge= most_frequent_pair,
            new_token_id=new_token_id
        )
        merges[most_frequent_pair] = new_token_id

        print(
            f"Merge {new_token_id}"
            f"{most_frequent_pair} -> {new_token_id}"
            f"| frequency: {frequency}"
            f"| tokens remaining: {len(token_ids)}"
        )
    return token_ids, merges

def build_vocabulary(merges):

    vocabulary = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    for pair, new_token_id in merges.items():  # .items() returns key-value pairs as tuples
        left_token_id, right_token_id = pair  # unpacks two tuple elements and store it in pair
        vocabulary[new_token_id] = (
            vocabulary[left_token_id]
            + vocabulary[right_token_id]
        )
    return vocabulary

def decode(
            token_ids,
            vocabulary,
            errors="strict"
    ):
    decoded_bytes = b"".join(
        vocabulary[token_id]
        for token_id in token_ids
    )
    decoded_text = decoded_bytes.decode(
        "utf-8",
        errors=errors
    )
    return decoded_text

def encode(text, merges):
    token_ids = list(text.encode("utf-8"))

    for pair_to_merge, new_token_id in merges.items():
        token_ids = merge_pair(
            token_ids = token_ids,
            pair_to_merge=pair_to_merge,
            new_token_id=new_token_id
        )
    return token_ids

def save_tokenizer(merges, vocabulary, output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )
    merge_data=[]

    for pair, new_token_id in merges.items():
        merge_data.append({
            "left_token_id": pair[0],
            "right_token_id": pair[1],
            "new_token_id": new_token_id
        })

    vocabulary_data = {}
    for token_id, token_bytes in vocabulary.items():
        vocabulary_data[str(token_id)] = list(token_bytes)
    merges_path = output_directory / "merges.json"
    vocabulary_path = output_directory / "vocabulary.json"

    merges_path.write_text(
        json.dumps(merge_data, indent=2),
        encoding="utf-8"
    )
    vocabulary_path.write_text(
        json.dumps(vocabulary_data, indent=2),
        encoding="utf-8"
    )

    print("\nTokenizer saved to: ")
    print(output_directory)

def load_tokenizer(output_directory):

    output_directory = Path(output_directory)
    merges_path = output_directory / "merges.json"
    vocabulary_path = output_directory / "vocabulary.json"

    merge_data = json.loads(
        merges_path.read_text(encoding="utf-8")
    )
    vocabulary_data = json.loads(
        vocabulary_path.read_text(encoding="utf-8")
    )

    merges= {}
    for rule in merge_data:
        pair = (
            rule["left_token_id"],
            rule["right_token_id"]
        )
        merges[pair] = rule["new_token_id"]

    vocabulary = {}
    for token_id, bytes_values in vocabulary_data.items():
        vocabulary[int(token_id)] = bytes(bytes_values)

    return merges, vocabulary

if __name__ == "__main__":

    training_text = "abababab"

    compressed_tokens, learned_merges = train_bpe(
        text=training_text,
        vocabulary_size=258
    )

    vocabulary = build_vocabulary(learned_merges)

    tokenizer_directory = "artifacts/tokenizer"

    # Save tokenizer
    save_tokenizer(
        merges=learned_merges,
        vocabulary=vocabulary,
        output_directory=tokenizer_directory
    )

    # Load tokenizer again
    loaded_merges, loaded_vocabulary = load_tokenizer(
        output_directory=tokenizer_directory
    )

    new_text = "abab"

    encoded_tokens = encode(
        text=new_text,
        merges=loaded_merges
    )

    decoded_text = decode(
        token_ids=encoded_tokens,
        vocabulary=loaded_vocabulary
    )

    print("\nLearned merges:")
    print(learned_merges)

    print("\nLoaded merges:")
    print(loaded_merges)

    print("\nNew text:")
    print(new_text)

    print("\nEncoded tokens:")
    print(encoded_tokens)

    print("\nDecoded text:")
    print(decoded_text)

    assert loaded_merges == learned_merges
    assert decoded_text == new_text

    print("\nSave-load round trip passed.")

"""
BPE basically first of all understand what are the repeated byte pairs in the text,, then replaces that repeated pair
with one extra token ID so the input stream gets shorter for model to learn on it"

multiple "ab", "ab" pairs having separate token IDs (1,2) for each pair, BPE will assign one unique and same token ID (256)
everytime this pair repeats. 
"""