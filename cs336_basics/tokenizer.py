from typing import Iterator, Iterable, Union
from cs336_basics.common import write_vocab_to_file, read_vocab_from_file, write_merges_to_file, read_merges_from_file
import regex as re
from cs336_basics.train_bpe import PAT
import torch
import numpy as np
import cProfile
import pstats

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens. This function accepts
        the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.merges_dict = {t : i for i, t in enumerate(merges)}
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        special_tokens = special_tokens if special_tokens is not None else []
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes in self.vocab.values():
                print(f"Special token '{token}' already exists in vocabulary.")
            else:
                self.vocab[len(self.vocab)] = token_bytes
                print(f"Added special token '{token}' to vocabulary with ID {len(self.vocab)-1}.")
        
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self.id_to_bytes = {k: v for k, v in self.vocab.items()}

        self.special_tokens_to_id = {}
        for token in sorted([token.encode('utf-8') for token in special_tokens], key=lambda x: len(x), reverse=True):
            self.special_tokens_to_id[token] = self.bytes_to_id[token]
    

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        vocab = read_vocab_from_file(vocab_filepath)
        merges = read_merges_from_file(merges_filepath)
        return cls(vocab, merges, special_tokens)


    def tokenize_special_tokens(self, text) -> list[Union[str, int]]:
        # Precompile regex patterns and decode tokens in advance
        patterns = []
        replacements = {}
        
        for special_token in self.special_tokens_to_id.keys():
            token_str = special_token.decode('utf-8')
            patterns.append(re.escape(token_str))
            replacements[token_str] = self.special_tokens_to_id[special_token]
        
        if len(patterns) == 0:
            return [text]
        # Create regex pattern to match all special tokens
        pattern = re.compile('|'.join(patterns))
        
        results = []
        last_end = 0
        
        # Use finditer to avoid repeated searching
        for match in pattern.finditer(text):
            # Add regular text before the match
            if match.start() > last_end:
                results.append(text[last_end:match.start()])
            
            # Add special token ID
            matched_text = match.group()
            results.append(replacements[matched_text])
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            results.append(text[last_end:])
        
        return results
  

    def merge_key(self, key: tuple[bytes], bytes_pair: tuple[bytes, bytes]) -> tuple[bytes]:
        i = 0
        L = len(key)
        result = []
        new_bytes = bytes_pair[0] + bytes_pair[1]
        while i < L:
            if i < L - 1 and key[i] == bytes_pair[0] and key[i + 1] == bytes_pair[1]:
                result.append(new_bytes)
                i += 2
            else:
                result.append(key[i])
                i += 1
        return tuple(result)


    def tokenize_word(self, word:str) -> list[int]:
        word_bytes = [bytes([x]) for x in word.encode('utf-8')]

        while True:
            merged = False
            rank = self.vocab_size
            best_pair = None
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                if pair in self.merges_dict and self.merges_dict[pair] < rank:
                    best_pair = pair
                    rank = self.merges_dict[pair]
                    merged = True
            if merged:
                word_bytes = self.merge_key(word_bytes, best_pair)
            else:
                break
        token_ids = [self.bytes_to_id[byte] for byte in word_bytes]
        return token_ids


    def tokenize_normal_text(self, text:bytes) -> list[int]:
        words = re.findall(PAT, text)
        token_ids = []
        for word in words:
            token_ids.extend(self.tokenize_word(word))
        return token_ids


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        # tokenize the text, find all special tokens, and replace them with their IDs
        results = self.tokenize_special_tokens(text)

        # the results in previous step will make the text into a list of Union[str, int]
        token_ids = []
        for elem in results:
            if isinstance(elem, str):
                token_ids.extend(self.tokenize_normal_text(elem))
            elif isinstance(elem, int):
                token_ids.append(elem)
            else:
                raise TypeError(f"Unexpected type {type(elem)} in results.")

        return token_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """
        remaining_text = ""
        for text in iterable:
            text = remaining_text + text
            match = re.search(PAT, text, flags=re.REVERSE)
            if match:
                text = text[:match.end()]
                remaining_text = text[match.end():]
                for id in self.encode(text):
                    yield id
            else:
                remaining_text = text
        # If there's any remaining text after the loop, tokenize it
        if remaining_text:
            for id in self.encode(remaining_text):
                yield id


    def decode(self, ids: list[int]) -> str: 
        """
        Decode a sequence of token IDs into text.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        elif not isinstance(ids, list):
            raise TypeError(f"Unsupported input type: {type(ids)}")
        
        tokens = [self.id_to_bytes[id] for id in ids]
        text = b"".join(tokens).decode('utf-8', errors='replace')
        return text


def test_encode():
    vocab = {0: b"<|endoftext|>", 1: b"<|startoftext|>", 2: b"ab", 3: b"a", 4: b"b", 5: b"c"}
    merges = [(b"a", b"b"), (b"ab", b"c")]
    special_tokens = ["<|endoftext|>", "<|startoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    results = tokenizer.encode("<|startoftext|>abacaba<|endoftext|>")
    print(tokenizer.encode("<|startoftext|>abacaba<|endoftext|>"))
    assert results == [1, 2, 3, 5, 2, 3, 0]


def tokenize_data(
    vocab_filepath:str,
    merges_filepath: str,
    train_filepath: str,
    eot_text: str,
):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=[eot_text]
    )
    # load training data
    text = ""
    with open(train_filepath, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        text = "".join(lines)

    eot_id = tokenizer.encode(eot_text)[0]
    assert eot_text == tokenizer.decode([eot_id]), "EOT_TEXT {EOT_TEXT} is not valid"

    # do tokenizer to training data
    token_ids = np.array(tokenizer.encode(text))
    token_ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
    return token_ids, eot_id


def test_encode_perf():
    # profiler = cProfile.Profile()
    # profiler.enable()
    tokenize_data(
        vocab_filepath="data/TinyStoriesV2-GPT4-vocab.txt",
        merges_filepath="data/TinyStoriesV2-GPT4-merges.txt",
        train_filepath="data/TinyStoriesV2-GPT4-valid.txt",
        eot_text="<|endoftext|>"
    )
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats(pstats.SortKey.TIME)  # 按耗时排序
    # stats.print_stats(10)  # 显示前10个最耗时的函数


if __name__ == "__main__":
    test_encode_perf()
