from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import os
from typing import BinaryIO

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


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))