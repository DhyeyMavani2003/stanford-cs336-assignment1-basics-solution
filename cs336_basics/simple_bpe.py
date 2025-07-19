#!/usr/bin/env python3
"""
Basic BPE (Byte-Pair Encoding) implementation for educational purposes.
This demonstrates the core concepts before building the full assignment solution.
"""

import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# GPT-2 style pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class SimpleBPE:
    def __init__(self, special_tokens: List[str] = None):
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self.vocab = {}
        self.merges = []
        
    def _split_on_special_tokens(self, text: str) -> List[str]:
        """Split text on special tokens to prevent cross-boundary merging"""
        if not self.special_tokens:
            return [text]
        
        import re as built_in_re
        pattern = "|".join(built_in_re.escape(token) for token in self.special_tokens)
        parts = built_in_re.split(f"({pattern})", text)
        return [part for part in parts if part]
    
    def _pretokenize(self, text: str) -> List[str]:
        """Apply regex-based pre-tokenization"""
        return re.findall(PAT, text)
    
    def _get_pairs(self, word: Tuple[bytes, ...]) -> List[Tuple[bytes, bytes]]:
        """Get all adjacent pairs in a word (represented as tuple of bytes)"""
        pairs = []
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i + 1]))
        return pairs
    
    def _count_pairs(self, word_freqs: Dict[Tuple[bytes, ...], int]) -> Counter:
        """Count all pairs across all words with their frequencies"""
        pair_counts = Counter()
        
        for word, freq in word_freqs.items():
            pairs = self._get_pairs(word)
            for pair in pairs:
                pair_counts[pair] += freq
        
        return pair_counts
    
    def _merge_pair(self, word: Tuple[bytes, ...], pair: Tuple[bytes, bytes], new_token: bytes) -> Tuple[bytes, ...]:
        """Merge a specific pair in a word"""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    
    def train(self, text: str, vocab_size: int = 300):
        """Train BPE tokenizer on the given text"""
        print(f"Training BPE tokenizer with vocab_size={vocab_size}")
        
        # Step 1: Initialize vocabulary with bytes and special tokens
        self.vocab = {}
        next_id = 0
        
        # Add special tokens first
        for token in self.special_tokens:
            self.vocab[token.encode('utf-8')] = next_id
            next_id += 1
        
        # Add all possible bytes (0-255)
        for i in range(256):
            byte_token = bytes([i])
            if byte_token not in self.vocab:  # Avoid duplicating special token bytes
                self.vocab[byte_token] = next_id
                next_id += 1
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Step 2: Split text and pre-tokenize
        text_parts = self._split_on_special_tokens(text)
        word_freqs = Counter()
        
        for part in text_parts:
            if part in self.special_tokens:
                # Special tokens are already in vocab, skip pre-tokenization
                continue
            else:
                # Pre-tokenize this part
                pretokens = self._pretokenize(part)
                for pretoken in pretokens:
                    # Convert to tuple of bytes
                    byte_tuple = tuple(pretoken.encode('utf-8'))
                    word_freqs[byte_tuple] += 1
        
        print(f"Number of unique pre-tokens: {len(word_freqs)}")
        print(f"Most common pre-tokens: {word_freqs.most_common(10)}")
        
        # Step 3: Perform BPE merges
        num_merges = vocab_size - len(self.vocab)
        print(f"Performing {num_merges} merges...")
        
        for merge_idx in range(num_merges):
            # Count all pairs
            pair_counts = self._count_pairs(word_freqs)
            
            if not pair_counts:
                print(f"No more pairs to merge after {merge_idx} merges")
                break
            
            # Find most frequent pair (with lexicographic tiebreaking)
            most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            freq = pair_counts[most_frequent_pair]
            
            print(f"Merge {merge_idx + 1}: {most_frequent_pair} (freq: {freq})")
            
            # Create new token by concatenating the pair
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            
            # Add to vocabulary
            self.vocab[new_token] = next_id
            next_id += 1
            
            # Record this merge
            self.merges.append(most_frequent_pair)
            
            # Update all words that contain this pair
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._merge_pair(word, most_frequent_pair, new_token)
                new_word_freqs[new_word] = freq
            
            word_freqs = new_word_freqs
        
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Number of merges performed: {len(self.merges)}")
    
    def show_vocab_sample(self, n=20):
        """Show a sample of the vocabulary"""
        print(f"\nVocabulary sample (showing first {n} items):")
        for i, (token, token_id) in enumerate(self.vocab.items()):
            if i >= n:
                break
            try:
                # Try to decode as string for display
                display = token.decode('utf-8', errors='replace')
                print(f"  {token_id:3d}: {token} -> '{display}'")
            except:
                print(f"  {token_id:3d}: {token} -> [binary]")

def main():
    # Test with the stylized example from the assignment
    print("="*60)
    print("BPE Training on Stylized Example")
    print("="*60)
    
    corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    
    bpe = SimpleBPE(special_tokens=["<|endoftext|>"])
    bpe.train(corpus, vocab_size=270)  # 256 bytes + 1 special + 13 merges
    bpe.show_vocab_sample(30)
    
    print(f"\nMerges performed:")
    for i, merge in enumerate(bpe.merges[:10]):  # Show first 10 merges
        byte1, byte2 = merge
        try:
            char1 = chr(byte1) if isinstance(byte1, int) else byte1.decode('utf-8', errors='replace')
            char2 = chr(byte2) if isinstance(byte2, int) else byte2.decode('utf-8', errors='replace')
            print(f"  {i+1:2d}. ({byte1}, {byte2}) -> '{char1}' + '{char2}'")
        except:
            print(f"  {i+1:2d}. {merge}")

if __name__ == "__main__":
    main()
