"""
PyTorch Implementation of X-Shape 24-Layer Transformer (0.1B SLM)
Author: DeepMind / Pair Programming Assistant
Architecture Highlights:
  - 24 Layers total (4 Outer-Head + 16 Core-Bottleneck + 4 Outer-Tail)
  - Variable-Width Hourglass / X-Shape parameter topology
  - Multi-scale Cross-Skip connection (Stage 1 -> Stage 3)
  - SwiGLU activations + Grouped Query Attention (GQA)
  - PyTorch 2.x F.scaled_dot_product_attention (SDPA / FlashAttention native)
"""

import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_x_shape import XShapeConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # cos, sin: [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1) # [b, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class XAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_dim: int,
        attention_dropout: float = 0.0,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        self.rope_dim = rope_dim
        self.dropout = attention_dropout
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = rotary_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            if self.sliding_window is not None and k.shape[2] > self.sliding_window:
                k = k[:, :, -self.sliding_window:, :]
                v = v[:, :, -self.sliding_window:, :]
        current_kv = (k, v)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Native PyTorch SDPA (auto-selects FlashAttention-2 or memory-efficient kernel)
        is_causal = attention_mask is None and q_len > 1
        if self.sliding_window is not None and q_len > 1 and attention_mask is None:
            # Construct sliding window causal mask (Gemma 2 style local attention)
            row = torch.arange(q_len, device=hidden_states.device).unsqueeze(1)
            col = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            sw_mask = (col <= row) & (row - col < self.sliding_window)
            attention_mask = sw_mask.unsqueeze(0).unsqueeze(0)
            is_causal = False

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask if not is_causal else None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
        output = self.o_proj(output)
        return output, current_kv


class XMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class XDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_dim: int,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.self_attn = XAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            rope_dim=rope_dim,
            attention_dropout=dropout,
            sliding_window=sliding_window,
        )
        self.mlp = XMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Self-Attention
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out, kv = self.self_attn(
            normed,
            rotary_emb=rotary_emb,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_out

        # Pre-LN SwiGLU MLP
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, kv


class XShapeModel(nn.Module):
    def __init__(self, config: XShapeConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size_outer)

        # RoPE Generators for Outer (head_dim=64) and Core (head_dim=60)
        head_dim_outer = config.hidden_size_outer // config.num_attention_heads_outer
        head_dim_core = config.hidden_size_core // config.num_attention_heads_core
        self.rotary_emb_outer = RotaryEmbedding(head_dim_outer, config.max_position_embeddings, config.rope_theta)
        self.rotary_emb_core = RotaryEmbedding(head_dim_core, config.max_position_embeddings, config.rope_theta)

        # -------------------------------------------------------------
        # Stage 1: Outer-Head Encoder (4 layers @ hidden 768)
        # -------------------------------------------------------------
        self.stage1_layers = nn.ModuleList([
            XDecoderLayer(
                hidden_size=config.hidden_size_outer,
                intermediate_size=config.intermediate_size_outer,
                num_heads=config.num_attention_heads_outer,
                num_kv_heads=config.num_key_value_heads_outer,
                rope_dim=head_dim_outer,
                rms_norm_eps=config.rms_norm_eps,
                dropout=config.attention_dropout_prob,
            )
            for _ in range(config.num_layers_outer_head)
        ])

        # -------------------------------------------------------------
        # Down-Bridge: 768 -> 480
        # -------------------------------------------------------------
        self.down_bridge = nn.Sequential(
            RMSNorm(config.hidden_size_outer, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size_outer, config.hidden_size_core, bias=False),
        )

        # -------------------------------------------------------------
        # Stage 2: Deep Core Bottleneck (16 layers @ hidden 480)
        # Interleaved Local-Global Attention (Gemma 2 style):
        # Even layers: Local Attention (Sliding Window = 512)
        # Odd layers:  Full Global Attention
        # -------------------------------------------------------------
        self.stage2_layers = nn.ModuleList([
            XDecoderLayer(
                hidden_size=config.hidden_size_core,
                intermediate_size=config.intermediate_size_core,
                num_heads=config.num_attention_heads_core,
                num_kv_heads=config.num_key_value_heads_core,
                rope_dim=head_dim_core,
                rms_norm_eps=config.rms_norm_eps,
                dropout=config.attention_dropout_prob,
                sliding_window=config.sliding_window if (i % 2 == 0) else None,
            )
            for i in range(config.num_layers_core)
        ])

        # -------------------------------------------------------------
        # Up-Bridge: 480 -> 768 + Cross-Skip Normalization & Learnable Gating
        # -------------------------------------------------------------
        self.up_bridge = nn.Sequential(
            RMSNorm(config.hidden_size_core, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size_core, config.hidden_size_outer, bias=False),
        )
        self.cross_skip_norm = RMSNorm(config.hidden_size_outer, eps=config.rms_norm_eps)
        self.cross_skip_gate = nn.Linear(config.hidden_size_outer, config.hidden_size_outer, bias=True)
        # Initialize gate weights to 0 so sigmoid(0) = 0.5 (balanced 50-50 flow at start)
        nn.init.zeros_(self.cross_skip_gate.weight)
        nn.init.zeros_(self.cross_skip_gate.bias)

        # -------------------------------------------------------------
        # Stage 3: Outer-Tail Decoder (4 layers @ hidden 768)
        # -------------------------------------------------------------
        self.stage3_layers = nn.ModuleList([
            XDecoderLayer(
                hidden_size=config.hidden_size_outer,
                intermediate_size=config.intermediate_size_outer,
                num_heads=config.num_attention_heads_outer,
                num_kv_heads=config.num_key_value_heads_outer,
                rope_dim=head_dim_outer,
                rms_norm_eps=config.rms_norm_eps,
                dropout=config.attention_dropout_prob,
            )
            for _ in range(config.num_layers_outer_tail)
        ])

        self.final_norm = RMSNorm(config.hidden_size_outer, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # 1. Token Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # 2. Rotary Embeddings
        rope_outer = self.rotary_emb_outer(hidden_states, seq_len)
        rope_core = self.rotary_emb_core(hidden_states, seq_len)

        new_kvs = []
        kv_idx = 0

        # -------------------------------------------------------------
        # Stage 1 Execution (4 layers)
        # -------------------------------------------------------------
        for layer in self.stage1_layers:
            past_kv = past_key_values[kv_idx] if past_key_values is not None else None
            hidden_states, kv = layer(hidden_states, rope_outer, position_ids, attention_mask, past_kv)
            new_kvs.append(kv)
            kv_idx += 1

        stage1_out = hidden_states  # Save for X-Cross Skip!

        # -------------------------------------------------------------
        # Down-Bridge: 768 -> 480
        # -------------------------------------------------------------
        hidden_states = self.down_bridge(hidden_states)

        # -------------------------------------------------------------
        # Stage 2 Execution (16 deep layers!)
        # -------------------------------------------------------------
        for layer in self.stage2_layers:
            past_kv = past_key_values[kv_idx] if past_key_values is not None else None
            hidden_states, kv = layer(hidden_states, rope_core, position_ids, attention_mask, past_kv)
            new_kvs.append(kv)
            kv_idx += 1

        # -------------------------------------------------------------
        # Up-Bridge: 480 -> 768
        # -------------------------------------------------------------
        hidden_states = self.up_bridge(hidden_states)

        # -------------------------------------------------------------
        # X-Cross Skip Connection: Learnable Cross-Gated residual
        # -------------------------------------------------------------
        gate = torch.sigmoid(self.cross_skip_gate(hidden_states))
        hidden_states = self.cross_skip_norm(hidden_states + gate * stage1_out)

        # -------------------------------------------------------------
        # Stage 3 Execution (4 layers)
        # -------------------------------------------------------------
        for layer in self.stage3_layers:
            past_kv = past_key_values[kv_idx] if past_key_values is not None else None
            hidden_states, kv = layer(hidden_states, rope_outer, position_ids, attention_mask, past_kv)
            new_kvs.append(kv)
            kv_idx += 1

        hidden_states = self.final_norm(hidden_states)
        return hidden_states, new_kvs


class XShapeForCausalLM(nn.Module):
    def __init__(self, config: XShapeConfig):
        super().__init__()
        self.config = config
        self.model = XShapeModel(config)
        self.lm_head = nn.Linear(config.hidden_size_outer, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        hidden_states, new_kvs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_kvs,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        curr = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(curr)
            logits = out["logits"][:, -1, :].clone()

            if repetition_penalty != 1.0:
                for token_id in set(curr[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            if temperature > 0.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)

            curr = torch.cat([curr, next_tok], dim=-1)
            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break
        return curr
