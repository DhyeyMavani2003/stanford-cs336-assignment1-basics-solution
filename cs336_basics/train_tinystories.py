#!/usr/bin/env python3
"""
Enhanced BPE Tokenizer with multiprocessing support for the TinyStories dataset.
This version includes parallelized pre-tokenization for better performance.
"""

import os
import regex as re
import time
import cProfile
import pstats
from typing import Dict, List, Tuple, Union
from collections import Counter
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pickle

# Import the pretokenization helper
import sys
sys.path.append('.')
from cs336_basics.pretokenization_example import find_chunk_boundaries

# GPT-2 style pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def process_chunk(args):
    """Process a single chunk of text for pre-tokenization."""
    chunk_text, special_tokens = args
    
    # Split on special tokens to prevent cross-boundary merging
    text_parts = _split_on_special_tokens(chunk_text, special_tokens)
    
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
                # Convert to tuple of bytes for easier manipulation
                byte_tuple = tuple(pretoken.encode('utf-8'))
                word_freqs[byte_tuple] += 1
    
    return word_freqs


def train_bpe_parallel(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = None,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer with parallel pre-tokenization.
    
    Args:
        input_path: Path to the text file containing training data
        vocab_size: Total desired vocabulary size (including special tokens and bytes)
        special_tokens: List of special tokens to add to vocabulary
        num_processes: Number of processes to use (default: CPU count)
        
    Returns:
        Tuple of (vocab, merges)
    """
    input_path = Path(input_path)
    
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Training BPE with {num_processes} processes...")
    
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
    
    print(f"Initial vocabulary size: {len(vocab)}")
    
    # Step 2: Parallel pre-tokenization
    print(f"Reading and chunking corpus from {input_path}...")
    
    # Find chunk boundaries based on special tokens
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8")
        )
    
    print(f"Created {len(boundaries)-1} chunks for processing")
    
    # Read chunks and prepare for parallel processing
    chunks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append((chunk_text, special_tokens))
    
    # Process chunks in parallel
    print("Pre-tokenizing chunks in parallel...")
    start_time = time.time()
    
    with Pool(num_processes) as pool:
        chunk_results = pool.map(process_chunk, chunks)
    
    # Combine results from all chunks
    word_freqs = Counter()
    for chunk_freq in chunk_results:
        word_freqs.update(chunk_freq)
    
    pretokenization_time = time.time() - start_time
    print(f"Pre-tokenization completed in {pretokenization_time:.2f} seconds")
    print(f"Found {len(word_freqs)} unique pre-tokens")
    
    # Step 3: Perform BPE merges (sequential, as this can't be easily parallelized)
    print("Starting BPE merge computation...")
    merge_start_time = time.time()
    
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
        
        # Record this merge
        merges.append(most_frequent_pair)
        
        # Create new merged token
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # Update word frequencies by merging the pair
        word_freqs = _merge_pair_in_words(word_freqs, most_frequent_pair, new_token)
        
        if (merge_idx + 1) % 100 == 0:
            print(f"Completed {merge_idx + 1} merges...")
    
    merge_time = time.time() - merge_start_time
    total_time = time.time() - start_time
    
    print(f"BPE merge computation completed in {merge_time:.2f} seconds")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Final vocab size: {len(vocab)}")
    
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


def analyze_vocabulary(vocab: Dict[int, bytes]) -> None:
    """Analyze the trained vocabulary."""
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS")
    print("="*60)
    
    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_length = len(longest_token)
    
    print(f"Longest token length: {longest_length} bytes")
    try:
        longest_str = longest_token.decode('utf-8', errors='replace')
        print(f"Longest token: {repr(longest_str)}")
    except:
        print(f"Longest token (bytes): {longest_token}")
    
    # Token length distribution
    length_dist = Counter(len(token) for token in vocab.values())
    print(f"\nToken length distribution:")
    for length in sorted(length_dist.keys()):
        print(f"  Length {length:2d}: {length_dist[length]:5d} tokens")
    
    # Show some example long tokens
    long_tokens = [(token_id, token) for token_id, token in vocab.items() if len(token) >= 10]
    long_tokens.sort(key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nTop 10 longest tokens:")
    for i, (token_id, token) in enumerate(long_tokens[:10]):
        try:
            token_str = token.decode('utf-8', errors='replace')
            print(f"  {i+1:2d}. ID {token_id:4d}: {len(token):2d} bytes - {repr(token_str)}")
        except:
            print(f"  {i+1:2d}. ID {token_id:4d}: {len(token):2d} bytes - {token}")


def main():
    """Train BPE on TinyStories and analyze results."""
    
    # Parameters for TinyStories training
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    print("Starting BPE training on TinyStories dataset...")
    print(f"Input: {input_path}")
    print(f"Target vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Train with profiling
    start_time = time.time()
    
    # Run with profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    vocab, merges = train_bpe_parallel(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    profiler.disable()
    total_time = time.time() - start_time
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges performed: {len(merges)}")
    
    # Save results
    vocab_path = "tinystories_vocab_10k.pkl"
    merges_path = "tinystories_merges_10k.pkl"
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)
    
    print(f"\nResults saved to:")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Merges: {merges_path}")
    
    # Analyze vocabulary
    analyze_vocabulary(vocab)
    
    # Show profiling results
    print(f"\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Show top 20 functions


if __name__ == "__main__":
    main()
