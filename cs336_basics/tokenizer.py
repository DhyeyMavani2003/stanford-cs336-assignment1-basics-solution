#!/usr/bin/env python3
"""
BPE (Byte-Pair Encoding) Tokenizer Training Implementation
for CS336 Assignment 1 - Problem 2 (train_bpe)

This module implements a byte-level BPE tokenizer training function
that matches the expected interface and output format.
"""

import os
import regex as re
from typing import Dict, List, Tuple, Union
from collections import Counter
from pathlib import Path


# GPT-2 style pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given corpus.
    
    Args:
        input_path: Path to the text file containing training data
        vocab_size: Total desired vocabulary size (including special tokens and bytes)
        special_tokens: List of special tokens to add to vocabulary
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: Dict mapping token IDs to their byte representations
        - merges: List of merge operations (pairs of bytes that were merged)
    """
    input_path = Path(input_path)
    
    # Step 1: Initialize vocabulary with special tokens and all bytes
    vocab = {}
    next_id = 0
    
    # Add special tokens first
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    
    # Add all 256 possible byte values
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    
    # Step 2: Read and pre-tokenize the corpus
    print(f"Reading corpus from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split on special tokens to prevent cross-boundary merging
    text_parts = _split_on_special_tokens(text, special_tokens)
    
    # Pre-tokenize and count word frequencies
    word_freqs = Counter()
    for part in text_parts:
        if part in special_tokens:
            # Skip special tokens during pre-tokenization
            continue
        else:
            # Apply regex pre-tokenization
            pretokens = re.findall(PAT, part)
            for pretoken in pretokens:
                # Convert to tuple of bytes objects (not individual byte values)
                byte_tuple = tuple(bytes([b]) for b in pretoken.encode('utf-8'))
                word_freqs[byte_tuple] += 1
    
    print(f"Found {len(word_freqs)} unique pre-tokens")
    
    # Step 3: Perform BPE merges
    num_merges = vocab_size - len(vocab)
    merges = []
    
    print(f"Performing {num_merges} BPE merges...")
    
    for merge_idx in range(num_merges):
        # Count all adjacent byte pairs
        pair_counts = _count_pairs(word_freqs)
        
        if not pair_counts:
            print(f"No more pairs to merge after {merge_idx} merges")
            break
        
        # Find most frequent pair (lexicographic tiebreaking)
        most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        # Record this merge (pairs should be bytes objects)
        merge_bytes = (most_frequent_pair[0], most_frequent_pair[1])
        merges.append(merge_bytes)
        
        # Create new merged token
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # Update word frequencies by merging the pair
        word_freqs = _merge_pair_in_words(word_freqs, most_frequent_pair, new_token)
        
        if (merge_idx + 1) % 50 == 0:
            print(f"Completed {merge_idx + 1} merges...")
    
    print(f"BPE training completed. Final vocab size: {len(vocab)}")
    
    return vocab, merges


def _split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """Split text on special tokens to prevent merging across boundaries."""
    if not special_tokens:
        return [text]
    
    # Create regex pattern with escaped special tokens
    import re as builtin_re
    pattern = "|".join(builtin_re.escape(token) for token in special_tokens)
    
    # Split but keep the separators
    parts = builtin_re.split(f"({pattern})", text)
    
    # Filter out empty strings
    return [part for part in parts if part]


def _count_pairs(word_freqs: Dict[Tuple[bytes, ...], int]) -> Counter:
    """Count all adjacent byte pairs across all words."""
    pair_counts = Counter()
    
    for word, freq in word_freqs.items():
        # Count adjacent pairs in this word
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
    
    return pair_counts


def _merge_pair_in_words(
    word_freqs: Dict[Tuple[bytes, ...], int], 
    pair: Tuple[bytes, bytes], 
    new_token: bytes
) -> Dict[Tuple[bytes, ...], int]:
    """Merge a specific pair in all words and return updated frequencies."""
    new_word_freqs = {}
    
    for word, freq in word_freqs.items():
        new_word = _merge_pair_in_word(word, pair, new_token)
        new_word_freqs[new_word] = freq
    
    return new_word_freqs


def _merge_pair_in_word(word: Tuple[bytes, ...], pair: Tuple[bytes, bytes], new_token: bytes) -> Tuple[bytes, ...]:
    """Merge a specific pair in a single word."""
    new_word = []
    i = 0
    
    while i < len(word):
        # Check if we can merge at this position
        if (i < len(word) - 1 and 
            word[i] == pair[0] and 
            word[i + 1] == pair[1]):
            # Merge the pair
            new_word.append(new_token)
            i += 2  # Skip both elements of the pair
        else:
            # Keep the current element
            new_word.append(word[i])
            i += 1
    
    return tuple(new_word)


# Example usage and testing
if __name__ == "__main__":
    # Test with a small corpus
    test_corpus = "Hello world! This is a test corpus for BPE training."
    
    # Write to temporary file
    test_file = "/tmp/test_corpus.txt"
    with open(test_file, 'w') as f:
        f.write(test_corpus)
    
    # Train BPE
    vocab, merges = train_bpe(
        input_path=test_file,
        vocab_size=300,  # 1 special + 256 bytes + 43 merges
        special_tokens=["<|endoftext|>"]
    )
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Show first few merges
    print(f"\nFirst 10 merges:")
    for i, (b1, b2) in enumerate(merges[:10]):
        try:
            char1 = chr(b1) if isinstance(b1, int) else b1.decode('utf-8', errors='replace')
            char2 = chr(b2) if isinstance(b2, int) else b2.decode('utf-8', errors='replace')
            print(f"  {i+1:2d}. ({b1}, {b2}) -> '{char1}' + '{char2}'")
        except:
            print(f"  {i+1:2d}. {(b1, b2)}")


class Tokenizer:
    """
    BPE (Byte-Pair Encoding) Tokenizer for encoding and decoding text.
    
    This tokenizer applies the learned BPE merges to encode text into token IDs
    and can decode token IDs back to text. It supports special tokens and 
    memory-efficient processing of large files.
    """
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        """
        Initialize the tokenizer with vocabulary, merges, and special tokens.
        
        Args:
            vocab: Dictionary mapping token IDs to their byte representations
            merges: List of BPE merge operations in order of creation
            special_tokens: Optional list of special tokens to add to vocabulary
        """
        # Copy the vocabulary to avoid modifying the original
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []
        
        # Create reverse vocabulary mapping (bytes -> token_id)
        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        
        # Handle special tokens
        if special_tokens:
            for special_token in special_tokens:
                special_bytes = special_token.encode('utf-8')
                if special_bytes not in self.byte_to_id:
                    # Add new special token to vocabulary
                    new_id = len(self.vocab)
                    self.vocab[new_id] = special_bytes
                    self.byte_to_id[special_bytes] = new_id
        
        # Create a more efficient merge lookup
        self.merge_rules = {}
        for i, (a, b) in enumerate(self.merges):
            self.merge_rules[(a, b)] = i
        
        # Create special token pattern for pre-tokenization if we have special tokens
        if self.special_tokens:
            # Sort special tokens by length (descending) to ensure greedy matching
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # Escape special characters in tokens and join with |
            escaped_tokens = [re.escape(token) for token in sorted_tokens]
            self.special_token_pattern = '(' + '|'.join(escaped_tokens) + ')'
        else:
            self.special_token_pattern = None
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        Create a tokenizer from saved vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to the pickled vocabulary file
            merges_filepath: Path to the pickled merges file  
            special_tokens: Optional list of special tokens
            
        Returns:
            Tokenizer instance
        """
        import pickle
        
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
            
        return cls(vocab, merges, special_tokens)
    
    def _apply_bpe(self, word_bytes: bytes) -> List[bytes]:
        """
        Apply BPE merges to a sequence of bytes representing a single pre-token.
        
        Args:
            word_bytes: Byte sequence for a single pre-token
            
        Returns:
            List of byte sequences after applying BPE merges
        """
        if len(word_bytes) <= 1:
            return [word_bytes]
        
        # Start with individual bytes
        pairs = []
        word = [bytes([b]) for b in word_bytes]
        
        # Repeatedly find and apply the highest priority merge
        while True:
            # Find all adjacent pairs
            pairs = []
            for i in range(len(word) - 1):
                pairs.append((word[i], word[i + 1], i))
            
            if not pairs:
                break
                
            # Find the pair with the highest merge priority (lowest index in merges list)
            best_pair = None
            best_merge_idx = float('inf')
            
            for a, b, pos in pairs:
                if (a, b) in self.merge_rules:
                    merge_idx = self.merge_rules[(a, b)]
                    if merge_idx < best_merge_idx:
                        best_merge_idx = merge_idx
                        best_pair = (a, b, pos)
            
            if best_pair is None:
                break
                
            # Apply the best merge
            a, b, pos = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i == pos:
                    new_word.append(a + b)
                    i += 2  # Skip the next element since we merged
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        return word
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into a sequence of token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        # First, split by special tokens if we have any
        if self.special_token_pattern:
            # Split text on special tokens, keeping the delimiters
            parts = re.split(self.special_token_pattern, text)
        else:
            parts = [text]
        
        token_ids = []
        for part in parts:
            if not part:  # Skip empty parts
                continue
                
            # Check if this part is a special token
            if part in self.special_tokens:
                special_bytes = part.encode('utf-8')
                if special_bytes in self.byte_to_id:
                    token_ids.append(self.byte_to_id[special_bytes])
                continue
            
            # Pre-tokenize this part using GPT-2 style regex
            pre_tokens = re.findall(PAT, part)
            
            for pre_token in pre_tokens:
                # Convert to UTF-8 bytes
                pre_token_bytes = pre_token.encode('utf-8')
                
                # Apply BPE to get list of byte sequences
                bpe_tokens = self._apply_bpe(pre_token_bytes)
                
                # Convert each byte sequence to token ID
                for token_bytes in bpe_tokens:
                    if token_bytes in self.byte_to_id:
                        token_ids.append(self.byte_to_id[token_bytes])
                    else:
                        # If we can't find the token, this shouldn't happen with proper BPE
                        # But handle gracefully by encoding individual bytes
                        for byte_val in token_bytes:
                            single_byte = bytes([byte_val])
                            if single_byte in self.byte_to_id:
                                token_ids.append(self.byte_to_id[single_byte])
        
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back to text.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if not ids:
            return ""
        
        # Concatenate all byte sequences
        byte_sequence = b""
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
            # If token_id not in vocab, skip it (could be invalid input)
        
        # Decode bytes to UTF-8 string, replacing invalid sequences
        try:
            return byte_sequence.decode('utf-8', errors='replace')
        except Exception:
            # Fallback if something goes wrong
            return byte_sequence.decode('utf-8', errors='replace')
    
    def encode_iterable(self, iterable):
        """
        Memory-efficient encoding of an iterable of strings.
        
        This method processes text chunks one at a time to avoid loading
        large files entirely into memory.
        
        Args:
            iterable: Iterable of strings (e.g., file handle, list of strings)
            
        Yields:
            Token IDs one at a time
        """
        for text_chunk in iterable:
            if isinstance(text_chunk, str):
                # Encode this chunk and yield each token ID
                token_ids = self.encode(text_chunk)
                for token_id in token_ids:
                    yield token_id
            else:
                # Handle bytes or other types by converting to string first
                try:
                    text_chunk = text_chunk.decode('utf-8', errors='replace')
                    token_ids = self.encode(text_chunk)
                    for token_id in token_ids:
                        yield token_id
                except AttributeError:
                    # If it's already a string or doesn't have decode method
                    token_ids = self.encode(str(text_chunk))
                    for token_id in token_ids:
                        yield token_id
