"""
Configuration for X-Shape 24-Layer Transformer (0.1B SLM)
Architecture:
  - Total Layers: 24 (Double the depth of standard 12-layer 0.1B)
  - Stage 1 (Entrance): 4 layers @ hidden 768 (FFN 2048)
  - Stage 2 (Bottleneck Core): 16 layers @ hidden 480 (FFN 1280)
  - Stage 3 (Exit Decoder): 4 layers @ hidden 768 (FFN 2048)
  - Cross-Skip Connection: X-residual from Stage 1 -> Stage 3
  - Vocab: 20,480 (Thai-20k Tokenizer)
  - Total Params: ~105.53M (0.105B)
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class XShapeConfig:
    vocab_size: int = 20480
    max_position_embeddings: int = 2048
    
    # Stage 1 & Stage 3 (Outer Wings: Wide)
    hidden_size_outer: int = 768
    intermediate_size_outer: int = 2048
    num_attention_heads_outer: int = 12
    num_key_value_heads_outer: int = 4
    num_layers_outer_head: int = 4   # Stage 1: Layers 0..3
    num_layers_outer_tail: int = 4   # Stage 3: Layers 20..23
    
    # Stage 2 (Core Bottleneck: Deep & Narrow)
    hidden_size_core: int = 480
    intermediate_size_core: int = 1344  # Expanded from 1280 (+1.47M parameters for extra factual capacity)
    num_attention_heads_core: int = 8
    num_key_value_heads_core: int = 2
    num_layers_core: int = 16        # Stage 2: Layers 4..19 (16 deep layers!)
    
    # Normalization & Regularization
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    
    # Total Layer Count Property
    @property
    def total_layers(self) -> int:
        return self.num_layers_outer_head + self.num_layers_core + self.num_layers_outer_tail
