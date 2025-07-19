#!/usr/bin/env python3
"""
Neural Network Building Blocks for CS336 Assignment 1 - Transformer Architecture
Implements basic components: Linear, Embedding, RMSNorm, etc.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class Linear(nn.Module):
    """
    Linear transformation module without bias.
    
    Performs y = W @ x where W is a learnable weight matrix.
    This follows the column vector convention used in the assignment.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the Linear module.
        
        Args:
            in_features: Size of input features (din)
            out_features: Size of output features (dout)  
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameter W of shape (out_features, in_features)
        # This stores W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights according to assignment specification
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters according to assignment specification."""
        # Linear weights: N(μ=0, σ²=2/(din+dout)) truncated at [-3σ, 3σ]
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Use @ operator for matrix multiplication
        # x @ W^T gives us the desired transformation
        return x @ self.weight.T
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'


class Embedding(nn.Module):
    """
    Token embedding module.
    
    Maps token IDs to dense vectors using manual indexing (no nn.Embedding).
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the Embedding module.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors (d_model)
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding weight parameter with shape (num_embeddings, embedding_dim)
        # Store with d_model (embedding_dim) as the final dimension
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights according to assignment specification
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters according to assignment specification."""
        # Embedding: N(μ=0, σ²=1) truncated at [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids: Token ID tensor of shape (...,)
            
        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        # Manual embedding lookup without using nn.functional.embedding
        # Index into the weight matrix to get embeddings
        return self.weight[token_ids]
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}'


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm normalizes activations using the root mean square and applies
    learnable scaling parameters. This is used in pre-norm Transformer blocks.
    """
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the RMSNorm module.
        
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # Learnable gain parameters (one per model dimension)
        self.weight = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
        
        # Initialize weights according to assignment specification
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters according to assignment specification."""
        # RMSNorm: Initialize to 1
        nn.init.ones_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        # Store original dtype for later restoration
        in_dtype = x.dtype
        
        # Upcast to float32 to prevent overflow when squaring
        x = x.to(torch.float32)
        
        # Calculate RMS along the last dimension (d_model)
        # RMS(a) = sqrt(1/d_model * sum(a_i^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        
        # Apply RMSNorm: x_i / RMS(x) * g_i
        normalized = x / rms
        
        # Apply learnable gain parameters
        result = normalized * self.weight
        
        # Return result in original dtype
        return result.to(in_dtype)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'd_model={self.d_model}, eps={self.eps}'


class SwiGLU(nn.Module):
    """
    SwiGLU position-wise feed-forward network.
    
    Combines SiLU (Swish) activation with Gated Linear Units (GLU).
    FFN(x) = W2(SiLU(W1x) ⊙ W3x)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the SwiGLU module.
        
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension of feed-forward network
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Three linear transformations as per SwiGLU formula
        # W1: d_model -> d_ff (for SiLU path)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # W2: d_ff -> d_model (output projection)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        # W3: d_model -> d_ff (for gating path)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward network.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        # Apply the three linear transformations
        w1_x = self.w1(x)    # (..., d_ff)
        w3_x = self.w3(x)    # (..., d_ff)
        
        # Apply SiLU (Swish) activation: SiLU(x) = x * sigmoid(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        
        # Element-wise multiplication (gating)
        gated = silu_w1_x * w3_x  # (..., d_ff)
        
        # Final linear transformation
        output = self.w2(gated)   # (..., d_model)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'd_model={self.d_model}, d_ff={self.d_ff}'


def make_swiglu_d_ff(d_model: int) -> int:
    """
    Calculate d_ff for SwiGLU such that it's approximately 8/3 * d_model
    but rounded to the nearest multiple of 64 for hardware efficiency.
    
    Args:
        d_model: Model dimension
        
    Returns:
        d_ff: Feed-forward dimension that's a multiple of 64
    """
    # Calculate 8/3 * d_model
    target_d_ff = int(8 * d_model / 3)
    
    # Round to nearest multiple of 64
    d_ff = ((target_d_ff + 31) // 64) * 64
    
    return d_ff


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network using SwiGLU.
    
    This is a convenience wrapper that automatically calculates d_ff
    and creates a SwiGLU module.
    """
    
    def __init__(
        self,
        d_model: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the position-wise feed-forward network.
        
        Args:
            d_model: Model dimension
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = make_swiglu_d_ff(d_model)
        
        # Create SwiGLU module
        self.swiglu = SwiGLU(d_model, self.d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        return self.swiglu(x)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'd_model={self.d_model}, d_ff={self.d_ff}'


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotary position embeddings to query and key vectors by rotating
    pairs of dimensions according to their position in the sequence.
    """
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the RoPE module.
        
        Args:
            theta: Base value for the rotation angles (Θ)
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length to precompute for
            device: Device to store buffers on
        """
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Ensure d_k is even (required for pair-wise rotations)
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        
        # Precompute the rotation angles for efficiency
        # θ_i,k = i * θ^(2k/d) for k ∈ {1, ..., d/2}
        self._precompute_rotations(device)
    
    def _precompute_rotations(self, device: Optional[torch.device]):
        """Precompute cos and sin values for all positions and dimensions."""
        
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        
        # Create dimension indices for pairs: [0, 1, 2, ..., d_k/2-1]
        dim_indices = torch.arange(self.d_k // 2, dtype=torch.float32, device=device)
        
        # Compute θ^(2k/d) for each dimension pair k
        # This gives us the base rotation angles for each dimension pair
        inv_freq = 1.0 / (self.theta ** (2 * dim_indices / self.d_k))
        
        # Compute all rotation angles: positions[:, None] * inv_freq[None, :]
        # Shape: (max_seq_len, d_k//2)
        angles = torch.outer(positions, inv_freq)
        
        # Precompute cos and sin values
        # Shape: (max_seq_len, d_k//2)
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        
        # Register as buffers (not parameters, since we don't want to learn them)
        self.register_buffer('cos_cached', cos_values, persistent=False)
        self.register_buffer('sin_cached', sin_values, persistent=False)
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            cos: Cosine values of shape (..., seq_len, d_k//2)
            sin: Sine values of shape (..., seq_len, d_k//2)
            
        Returns:
            Rotated tensor of same shape as input
        """
        # Split x into pairs: x_1, x_2, x_3, x_4, ... -> (x_1, x_2), (x_3, x_4), ...
        # Shape: (..., seq_len, d_k//2)
        x1 = x[..., 0::2]  # Even indices: 0, 2, 4, ...
        x2 = x[..., 1::2]  # Odd indices: 1, 3, 5, ...
        
        # Apply rotation:
        # [x1']   [cos -sin] [x1]   [x1*cos - x2*sin]
        # [x2'] = [sin  cos] [x2] = [x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave the rotated pairs back together
        # We need to combine (x1', x2') back into the original format
        result = torch.empty_like(x)
        result[..., 0::2] = rotated_x1  # Even indices
        result[..., 1::2] = rotated_x2  # Odd indices
        
        return result
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)
            
        Returns:
            Rotated tensor of same shape as input
        """
        # Get the sequence length
        seq_len = x.shape[-2]
        
        # Extract cos and sin values for the specified positions
        # token_positions shape: (..., seq_len)
        # cos_cached/sin_cached shape: (max_seq_len, d_k//2)
        
        # Index into our precomputed cos/sin arrays using token_positions
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k//2)
        
        # Apply the rotary position embedding
        return self._apply_rotary_pos_emb(x, cos, sin)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}'


def softmax(input: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax operation on the specified dimension with numerical stability.
    
    Uses the trick of subtracting the maximum value in the specified dimension
    to avoid numerical overflow issues where exp(vi) can become inf.
    
    Args:
        input: Input tensor of arbitrary shape
        dim: Dimension to apply softmax over
        
    Returns:
        Tensor of same shape as input with softmax applied over the specified dimension
    """
    # For numerical stability, subtract the max value along the specified dimension
    # This keeps the largest exponent at 0, preventing overflow
    input_max = torch.max(input, dim=dim, keepdim=True)[0]
    input_stable = input - input_max
    
    # Compute exp of the numerically stable input
    exp_values = torch.exp(input_stable)
    
    # Compute the sum along the specified dimension
    exp_sum = torch.sum(exp_values, dim=dim, keepdim=True)
    
    # Return the normalized probabilities
    return exp_values / exp_sum


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss with numerical stability.
    
    Computes the cross-entropy loss ℓ = -log(softmax(logits)[targets])
    for each example, then returns the average across the batch.
    
    Uses numerical stability tricks:
    - Subtracts the maximum logit value to prevent overflow
    - Cancels log and exp where possible to avoid computing full softmax
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size) containing unnormalized logits
        targets: Tensor of shape (batch_size,) containing target class indices
        
    Returns:
        Scalar tensor containing the average cross-entropy loss across the batch
    """
    batch_size, vocab_size = logits.shape
    
    # For numerical stability, subtract the max logit from each row
    # This prevents overflow when computing exp
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    logits_stable = logits - max_logits
    
    # Compute log-sum-exp for the denominator
    # log(sum(exp(logits_stable))) = log(sum(exp(logits - max_logits)))
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=1))
    
    # Get the logits for the target classes
    # logits_stable[i, targets[i]] gives the numerator logit for example i
    target_logits = logits_stable[torch.arange(batch_size), targets]
    
    # Compute cross-entropy loss for each example
    # loss = -log(softmax(logits)[target]) = -log(exp(logit_target) / sum(exp(logits)))
    # = -logit_target + log(sum(exp(logits)))
    # Since we subtracted max_logits, this becomes:
    # = -(logit_target - max_logit) + log(sum(exp(logits - max_logit)))
    # = -logit_target + max_logit + log_sum_exp
    # But max_logit cancels out since target_logits already has max_logit subtracted
    losses = -target_logits + log_sum_exp
    
    # Return the average loss across the batch
    return torch.mean(losses)


def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Implement scaled dot-product attention as described in Vaswani et al. [2017].
    
    Computes Attention(Q, K, V) = softmax(Q^T K / sqrt(d_k)) V
    
    Args:
        Q: Query tensor of shape (..., n_queries, d_k)
        K: Key tensor of shape (..., n_keys, d_k) 
        V: Value tensor of shape (..., n_keys, d_v)
        mask: Optional boolean mask of shape (..., n_queries, n_keys)
              True means attend, False means mask out (set to -inf before softmax)
              
    Returns:
        Output tensor of shape (..., n_queries, d_v)
    """
    # Get the dimension of keys for scaling
    d_k = Q.shape[-1]
    
    # Compute attention scores: Q^T @ K / sqrt(d_k)
    # Q shape: (..., n_queries, d_k)
    # K shape: (..., n_keys, d_k)
    # We want (..., n_queries, n_keys)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Convert boolean mask to float mask
        # True -> 0 (attend), False -> -inf (mask out)
        float_mask = torch.where(mask, 0.0, float('-inf'))
        scores = scores + float_mask
    
    # Apply softmax to get attention weights
    # Softmax over the last dimension (keys dimension)
    attention_weights = softmax(scores, dim=-1)
    
    # Apply attention weights to values
    # attention_weights shape: (..., n_queries, n_keys)
    # V shape: (..., n_keys, d_v)
    # Output shape: (..., n_queries, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module with causal masking and optional RoPE.
    
    Implements the multi-head attention mechanism from "Attention Is All You Need"
    (Vaswani et al., 2017) with causal masking for autoregressive language modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        use_rope: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Dimensionality of the model embeddings
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for RoPE precomputation
            theta: RoPE theta parameter
            use_rope: Whether to apply rotary position embeddings
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k = d_v = d_model / h
        self.d_v = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        # Projection layers for Q, K, V
        self.q_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype) 
        self.v_proj = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        
        # Output projection
        self.o_proj = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        
        # RoPE if enabled
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                theta=theta,
                device=device
            )
        else:
            self.rope = None
            
        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
            persistent=False
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position tensor of shape (..., seq_len)
                           If not provided and RoPE is enabled, uses default positions
                           
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        
        # Project to Q, K, V
        # Each has shape (..., seq_len, num_heads * d_k/d_v)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape and transpose to get (..., num_heads, seq_len, d_k/d_v)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
        
        # Apply RoPE if enabled
        if self.use_rope and self.rope is not None:
            if token_positions is None:
                # Default to sequential positions
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
                # Expand to match batch dimensions
                for _ in range(len(batch_shape)):
                    token_positions = token_positions.unsqueeze(0)
                token_positions = token_positions.expand(*batch_shape, seq_len)
            
            # Apply RoPE to Q and K (not V)
            # RoPE expects (..., seq_len, d_k), so we need to handle the head dimension
            # Reshape to treat heads as part of batch dimension
            q_shape = Q.shape
            k_shape = K.shape
            
            Q_for_rope = Q.reshape(-1, seq_len, self.d_k)
            K_for_rope = K.reshape(-1, seq_len, self.d_k)
            
            # Expand token_positions to match the head dimension
            pos_for_rope = token_positions.unsqueeze(-2).expand(*batch_shape, self.num_heads, seq_len)
            pos_for_rope = pos_for_rope.reshape(-1, seq_len)
            
            Q_for_rope = self.rope(Q_for_rope, pos_for_rope)
            K_for_rope = self.rope(K_for_rope, pos_for_rope)
            
            Q = Q_for_rope.reshape(q_shape)
            K = K_for_rope.reshape(k_shape)
        
        # Create causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
        # Convert to attention mask format (True = attend, False = mask out)
        causal_mask = ~mask  # Invert: False where we should mask, True where we should attend
        
        # Expand mask to match batch and head dimensions
        # Start with shape (seq_len, seq_len)
        # Add dimensions for batch and head: (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Now expand to match the actual batch and head dimensions
        # Target shape: (*batch_shape, num_heads, seq_len, seq_len)
        target_shape = (*batch_shape, self.num_heads, seq_len, seq_len)
        causal_mask = causal_mask.expand(target_shape)
        
        # Apply scaled dot-product attention
        # Q, K, V shapes: (..., num_heads, seq_len, d_k/d_v)
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        
        # Reshape back to (..., seq_len, num_heads * d_v)
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*batch_shape, seq_len, self.num_heads * self.d_v)
        
        # Apply output projection
        output = self.o_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with multi-head self-attention and feed-forward network.
    
    Implements the architecture:
    1. y1 = x + MultiHeadSelfAttention(RMSNorm(x))
    2. y2 = y1 + FeedForward(RMSNorm(y1))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Dimensionality of the model embeddings
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward inner layer
            max_seq_len: Maximum sequence length for RoPE
            theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Layer normalization for attention sublayer
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Multi-head self-attention with RoPE
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True,
            device=device,
            dtype=dtype
        )
        
        # Layer normalization for feed-forward sublayer
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Transformer block.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position tensor for RoPE
            
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # First sublayer: Multi-head self-attention with residual connection
        # y1 = x + MultiHeadSelfAttention(RMSNorm(x))
        attn_input = self.ln1(x)
        attn_output = self.attn(attn_input, token_positions)
        y1 = x + attn_output
        
        # Second sublayer: Feed-forward with residual connection
        # y2 = y1 + FeedForward(RMSNorm(y1))
        ffn_input = self.ln2(y1)
        ffn_output = self.ffn(ffn_input)
        y2 = y1 + ffn_output
        
        return y2


class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model.
    
    Implements the full architecture from token embeddings through multiple
    Transformer blocks to final language modeling head.
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Transformer Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            context_length: Maximum context length
            d_model: Model dimensionality
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward dimensionality
            theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        
        # Token embeddings
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Language modeling head (output projection to vocabulary)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer Language Model.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            token_positions: Optional position indices for RoPE
            
        Returns:
            Logits over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create default token positions if not provided
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            token_positions = token_positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # Final layer normalization
        x = self.ln_final(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer implementation as described in Loshchilov and Hutter (2019).
    
    AdamW decouples weight decay from the gradient-based update, which improves
    regularization compared to standard Adam.
    
    The algorithm maintains exponential moving averages of the gradient (first moment)
    and the squared gradient (second moment) for each parameter.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (α in the algorithm)
            betas: Coefficients (β1, β2) for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient (λ in the algorithm)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            The loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state["step"]
                # Compute bias-corrected second raw moment estimate
                bias_correction2 = 1 - beta2 ** state["step"]
                
                # Compute adjusted learning rate
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Apply weight decay
                p.data.mul_(1 - lr * weight_decay)
        
        return loss


def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.
    
    Args:
        it: Current iteration number
        max_learning_rate: Maximum learning rate (α_max)
        min_learning_rate: Minimum learning rate (α_min)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Number of cosine annealing iterations (T_c)
    
    Returns:
        Learning rate at iteration it
    """
    if it < warmup_iters:
        # Warmup phase: linear increase from 0 to max_learning_rate
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing phase
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(progress * math.pi)
        )
    else:
        # Post-annealing phase: constant minimum learning rate
        return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    Clip gradients to have L2 norm at most max_l2_norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_l2_norm: Maximum L2 norm for gradients
    """
    eps = 1e-6  # PyTorch default for numerical stability
    
    # Collect all gradients
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)
    
    if not gradients:
        return
    
    # Compute total L2 norm of all gradients
    total_norm = 0.0
    for grad in gradients:
        total_norm += grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    
    # Apply clipping if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in gradients:
            grad.mul_(clip_coef)


def get_batch(dataset, batch_size: int, context_length: int, device: str):
    """
    Sample a batch of input sequences and their corresponding targets from the dataset.
    
    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: Number of sequences in the batch
        context_length: Length of each sequence
        device: PyTorch device string (e.g., 'cpu', 'cuda:0', 'mps')
    
    Returns:
        Tuple of (input_sequences, target_sequences) where both are torch.LongTensor
        of shape (batch_size, context_length)
    """
    import numpy as np
    
    # Calculate the maximum valid starting index
    max_start_idx = len(dataset) - context_length
    
    # Randomly sample starting indices
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Create input and target sequences
    input_sequences = []
    target_sequences = []
    
    for start_idx in start_indices:
        # Input sequence: [start_idx, start_idx + context_length)
        input_seq = dataset[start_idx:start_idx + context_length]
        # Target sequence: [start_idx + 1, start_idx + context_length + 1)
        target_seq = dataset[start_idx + 1:start_idx + context_length + 1]
        
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    # Convert to numpy arrays and then to tensors
    input_array = np.array(input_sequences)
    target_array = np.array(target_sequences)
    
    # Convert to PyTorch tensors and move to specified device
    input_tensor = torch.from_numpy(input_array).long().to(device)
    target_tensor = torch.from_numpy(target_array).long().to(device)
    
    return input_tensor, target_tensor


def save_checkpoint(model, optimizer, iteration, out):
    """
    Save model, optimizer, and iteration state to a checkpoint file.
    
    Args:
        model: torch.nn.Module to save
        optimizer: torch.optim.Optimizer to save
        iteration: int iteration number
        out: str | os.PathLike | BinaryIO | IO[bytes] - destination to save to
    """
    # Create checkpoint dictionary with all necessary state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # Save to file or file-like object
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """
    Load model, optimizer, and iteration state from a checkpoint file.
    
    Args:
        src: str | os.PathLike | BinaryIO | IO[bytes] - source to load from
        model: torch.nn.Module to restore state to
        optimizer: torch.optim.Optimizer to restore state to
    
    Returns:
        int: The iteration number from the checkpoint
    """
    # Load checkpoint from file or file-like object
    checkpoint = torch.load(src)
    
    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return the iteration number
    return checkpoint['iteration']


def temperature_scaled_softmax(logits: torch.Tensor, temperature: float = 1.0, dim: int = -1) -> torch.Tensor:
    """
    Apply temperature scaling to logits and then softmax.
    
    Args:
        logits: Input logits tensor
        temperature: Temperature parameter (τ). Lower values make distribution more peaked.
        dim: Dimension to apply softmax over
    
    Returns:
        Temperature-scaled softmax probabilities
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Apply softmax
    return torch.softmax(scaled_logits, dim=dim)


def top_p_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to a probability distribution.
    
    Args:
        probs: Probability distribution of shape (..., vocab_size)
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Modified probability distribution with low-probability tokens set to 0
    """
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    
    # If p is 1.0, return the original distribution
    if p >= 1.0:
        return probs
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices where cumulative probability exceeds p
    # We want to keep tokens until we reach p
    cutoff_mask = cumulative_probs > p
    
    # Always keep at least the most probable token
    cutoff_mask[..., 0] = False
    
    # Set probabilities of tokens beyond cutoff to 0
    sorted_probs[cutoff_mask] = 0.0
    
    # Scatter back to original order
    result = torch.zeros_like(probs)
    result.scatter_(-1, sorted_indices, sorted_probs)
    
    # Renormalize to ensure it's a valid probability distribution
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-10)
    
    return result


def sample_from_distribution(probs: torch.Tensor) -> int:
    """
    Sample a token from a probability distribution.
    
    Args:
        probs: Probability distribution of shape (vocab_size,)
    
    Returns:
        Sampled token index
    """
    return torch.multinomial(probs, 1).item()


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str = "",
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    stop_token: str = "<|endoftext|>"
) -> str:
    """
    Generate text from a language model using various sampling strategies.
    
    Args:
        model: Trained TransformerLM model
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Initial text prompt to continue from
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (lower = more deterministic)
        top_p: Top-p sampling threshold (1.0 = no filtering)
        device: Device to run generation on
        stop_token: Token to stop generation at
    
    Returns:
        Generated text including the prompt
    """
    model.eval()
    
    # Tokenize the prompt
    if prompt:
        input_tokens = tokenizer.encode(prompt)
    else:
        input_tokens = []
    
    # Convert to tensor
    generated_tokens = input_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Prepare input sequence (use last context_length tokens)
            context_length = model.context_length
            current_tokens = generated_tokens[-context_length:]
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(input_tensor)  # Shape: (1, seq_len, vocab_size)
            
            # Get logits for the last token (next token prediction)
            next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
            
            # Apply temperature scaling
            probs = temperature_scaled_softmax(next_token_logits, temperature)
            
            # Apply top-p sampling if specified
            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)
            
            # Sample next token
            next_token = sample_from_distribution(probs)
            
            # Add to generated sequence
            generated_tokens.append(next_token)
            
            # Check for stop token
            try:
                decoded_token = tokenizer.decode([next_token])
                if stop_token in decoded_token:
                    break
            except:
                # If decoding fails, continue (might be a partial token)
                pass
    
    # Decode the full sequence
    return tokenizer.decode(generated_tokens)


def generate_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    stop_token: str = "<|endoftext|>"
) -> list[str]:
    """
    Generate text for multiple prompts in parallel.
    
    Args:
        model: Trained TransformerLM model
        tokenizer: Tokenizer for encoding/decoding text
        prompts: List of text prompts to continue from
        max_tokens: Maximum number of tokens to generate per prompt
        temperature: Temperature for sampling
        top_p: Top-p sampling threshold
        device: Device to run generation on
        stop_token: Token to stop generation at
    
    Returns:
        List of generated texts
    """
    # For simplicity, process each prompt individually
    # A more efficient implementation would batch process when possible
    results = []
    for prompt in prompts:
        result = generate_text(
            model, tokenizer, prompt, max_tokens, 
            temperature, top_p, device, stop_token
        )
        results.append(result)
    
    return results
