# Unicode and Tokenization Problems — Solutions

---

## Problem (unicode1): Understanding Unicode — Answers

### (a) What Unicode character does `chr(0)` return?
`chr(0)` returns the NULL character (also known as NUL), which is represented as `'\x00'` in Python's string representation.

### (b) How does this character's string representation (`repr()`) differ from its printed representation?
- `repr()`: Shows `'\x00'` (hexadecimal escape sequence).
- Printed: Appears as nothing visible (an invisible character).

### (c) What happens when this character occurs in text?
- Acts as an invisible separator.
- Does not affect visual appearance when printed.
- Counted in string length and detectable programmatically.

---

## Problem (unicode2): Unicode Encodings — Answers

### (a) Why prefer UTF-8 for tokenizer training over UTF-16 or UTF-32?
- More space-efficient for ASCII and common characters.
- Variable-length encoding reduces overhead.
- Example: 
  - "hello" in UTF-8 = 5 bytes  
  - UTF-16 = 12 bytes  
  - UTF-32 = 24 bytes

### (b) Why is the incorrect decode function wrong? Example?
- `decode_utf8_bytes_to_str_wrong` decodes bytes individually.
- UTF-8 uses multi-byte sequences for non-ASCII — decoding partial sequences fails.

**Failing Example**: `"こんにちは".encode("utf-8")`  
Bytes: `[227, 129, 147, ...]`  
Each Japanese character takes 3 bytes — decoding `227` alone throws `UnicodeDecodeError`.

### (c) Invalid 2-byte sequence?
- `bytes([255, 254])` — Invalid UTF-8 start bytes.
- UTF-8 never uses 0xFF or 0xFE.

---

## Problem (train_bpe): BPE Tokenizer Training  
*(Implemented as a `.py` file)*

---

## Problem (train_bpe_tinystories): Results & Answers

### (a) Training Results

- Duration: **27.00 minutes**
- Memory: **≈11GB RAM**
- Longest Token: `' accomplishment'` (15 bytes) — meaningful complete word.

### (b) Profiling Analysis

- **Main Bottlenecks**:  
  - `_merge_pair_in_word`: 610.7s  
  - `_count_pairs`: 275.2s  
- Regex tokenization: Only 88.6s

### Key Insights

- **Vocabulary Composition**:
  - Long tokens are complete words.
  - Reduced 59,933 pre-tokens to 10,000 via 9,743 merges.

- **Performance**:
  - 11.3B function calls.
  - Merge operations dominate time.
  - Memory usage reasonable (~2.1GB).

- **Validation**:
  - BPE effectively learns subword units.
  - Long tokens match frequent words.
  - Balanced compression and linguistic value.

---

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText  
*(Implemented as a `.py` file)*  
*Details omitted.*

---

## Problem (transformer_accounting): Transformer LM Resource Accounting

### (a) GPT-2 XL Parameters & Memory

**Configuration**:
- `vocab_size`: 50,257  
- `context_length`: 1,024  
- `num_layers`: 48  
- `d_model`: 1,600  
- `num_heads`: 25  
- `d_ff`: 6,400  

**Result**:  
- Trainable Parameters: **≈2.1B**  
- Memory for params: **≈7.92 GB**

---

### (b) FLOPs Breakdown

- **Embeddings**: 164.7B FLOPs  
- **Multi-Head Attention**: 15.7B FLOPs  
- **Attention Computation**: 6.7B FLOPs  
- **Output Projection**: 5.24B FLOPs  
- **FFN (SwiGLU)**: ~62.9B FLOPs  
- **LM Head**: 164.7B FLOPs  
- **Total Forward Pass FLOPs**: **≈4.68 trillion**

---

### (c) FLOP-intensive Component

- **FFN**: 64.6%  
- **Attention**: 28.4%  
- **Embeddings/LM Head**: 3.5% each

---

### (d) Scaling Analysis

| Model         | Total FLOPs | Attention % | FFN % | Embeddings/LM Head % |
|---------------|-------------|-------------|-------|------------------------|
| GPT-2 Small   | 428.7B      | 22.5%       | 40.6% | 18.4% each             |
| GPT-2 Medium  | 1,138.5B    | 27.2%       | 54.3% | 9.3% each              |
| GPT-2 Large   | 2,389.5B    | 28.3%       | 60.7% | 5.5% each              |
| GPT-2 XL      | 4,678.0B    | 28.4%       | 64.6% | 3.5% each              |

---

### (e) Extended Context (16k tokens)

- FLOPs Increase:  
  - Original: 4.68T  
  - Extended: 152.15T → **32.5× increase**

- **Component Shifts**:
  - Attention: **→ 64.8%**
  - FFN: **↓ to 31.8%**
  - Embeddings/LM Head: **↓ to 1.7%**

---

## Problem (learning_rate_tuning): Tuning the Learning Rate

- `lr=1e1` (10): Converges slowly.
- `lr=1e2` (100): Rapid convergence; near zero loss at iteration 5.
- `lr=1e3` (1000): Divergence; loss explodes to `2.24×10^18`.

Conclusion: Higher LR speeds up convergence but too high leads to instability.

---

## Problem (adamw): Implement AdamW  
*(Implemented as a `.py` file)*

---

## Problem (adamwAccounting): Resource Accounting for AdamW

### (a) Peak Memory Usage Formula

- Parameters: `4 * (2VD + L(12D² + 2D) + D)`  
- Gradients: Same as parameters  
- AdamW State: `8 * (2VD + L(12D² + 2D) + D)`  
- Activations: `4 * batch_size * [L(16CD + 2C²) + CD + 2CV]`

*(V: vocab_size, C: context_length, L: num_layers, D: d_model)*

---

### (b) GPT-2 XL Memory Estimate

- GPU: 80GB  
- Memory usage: `5.85 × batch_size + 26.17 GB`  
- Max batch size: **9**

---

### (c) AdamW FLOPs

- Each optimizer step: **12 FLOPs per parameter**

---

### (d) Training Time Estimate

- **Config**: GPT-2 XL, batch size = 1024, steps = 400,000, A100 GPU @ 50% MFU  
- **Time**: ~**4,886.0 days**

---

