from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

def write_vocab_to_file(vocab, file_path):
    gpt2_byte_decoder = {k: v for k, v in gpt2_bytes_to_unicode().items()}
    with open(file_path, "w", encoding="utf-8") as f:
        for k, v in vocab.items():
            f.write(str(k) + " " + "".join([gpt2_byte_decoder[token] for token in v]))
            f.write("\n")


def read_vocab_from_file(file_path):
    gpt2_byte_encoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(file_path, "r", encoding='utf-8') as f:
        data = f.readlines()
    vocab = {}
    for line in data:
        index, token = line.strip().split(" ", 1)
        index = int(index)
        token_in_bytes = bytes([gpt2_byte_encoder[ch] for ch in token])
        vocab[index] = token_in_bytes
    return vocab


def write_merges_to_file(merges, file_path):
    gpt2_byte_decoder = {k: v for k, v in gpt2_bytes_to_unicode().items()}
    human_readable_merges = [
        (
            "".join([gpt2_byte_decoder[token] for token in merge_token_1]),
            "".join([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in merges
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for (a, b) in human_readable_merges:
            f.write(a + " " + b)
            f.write("\n")


def read_merges_from_file(file_path):
    gpt2_byte_encoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(file_path, "r", encoding='utf-8') as f:
        data = f.readlines()
    merges = []
    for line in data:
        token1, token2 = line.strip().split(" ", 1)
        tokens_in_bytes = bytes([gpt2_byte_encoder[ch] for ch in token1]), bytes([gpt2_byte_encoder[ch] for ch in token2])
        merges.append((tokens_in_bytes[0], tokens_in_bytes[1]))
    return merges