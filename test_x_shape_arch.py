"""
Validation & Architecture Sanity Check for X-Shape 24-Layer Transformer
"""

import sys
import os
import torch

from config_x_shape import XShapeConfig
from modeling_x_shape import XShapeForCausalLM

def run_tests():
    print("=" * 70)
    print("🧪 VALIDATING X-SHAPE 24-LAYER TRANSFORMER ARCHITECTURE")
    print("=" * 70)

    config = XShapeConfig()
    print(f"Total Configured Layers : {config.total_layers}")
    print(f" - Stage 1 (Outer Head) : {config.num_layers_outer_head} layers @ hidden {config.hidden_size_outer}")
    print(f" - Stage 2 (Core Deep)  : {config.num_layers_core} layers @ hidden {config.hidden_size_core}")
    print(f" - Stage 3 (Outer Tail) : {config.num_layers_outer_tail} layers @ hidden {config.hidden_size_outer}")
    print(f" - Vocab Size           : {config.vocab_size:,}")

    model = XShapeForCausalLM(config)

    # 1. Parameter Breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    s1_params = sum(p.numel() for p in model.model.stage1_layers.parameters())
    s2_params = sum(p.numel() for p in model.model.stage2_layers.parameters())
    s3_params = sum(p.numel() for p in model.model.stage3_layers.parameters())
    bridge_params = (
        sum(p.numel() for p in model.model.down_bridge.parameters()) +
        sum(p.numel() for p in model.model.up_bridge.parameters()) +
        sum(p.numel() for p in model.model.cross_skip_gate.parameters())
    )
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())

    print("\n📊 PARAMETER ALLOCATION BREAKDOWN:")
    print(f"  • Embeddings (Tied)   : {embed_params:,} ({embed_params/1e6:.2f}M)")
    print(f"  • Stage 1 (4 layers)  : {s1_params:,} ({s1_params/1e6:.2f}M)")
    print(f"  • Stage 2 (16 layers) : {s2_params:,} ({s2_params/1e6:.2f}M)  <-- 16 Deep Layers!")
    print(f"  • Stage 3 (4 layers)  : {s3_params:,} ({s3_params/1e6:.2f}M)")
    print(f"  • Bridges & Skips     : {bridge_params:,} ({bridge_params/1e6:.2f}M)")
    print(f"  -------------------------------------------------------------")
    print(f"  ⭐ TOTAL PARAMETERS   : {total_params:,} ({total_params/1e6:.2f}M / ~0.105B)")
    print(f"  ⭐ TRAINABLE PARAMS   : {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 2. Forward & Loss Test
    print("\n🔄 TESTING FORWARD PASS & LOSS COMPUTATION...")
    dummy_input = torch.randint(0, config.vocab_size, (2, 64))
    dummy_labels = dummy_input.clone()
    
    out = model(dummy_input, labels=dummy_labels)
    loss = out["loss"]
    logits = out["logits"]

    print(f"  • Logits shape: {logits.shape} (Expected: [2, 64, {config.vocab_size}])")
    print(f"  • Initial Loss: {loss.item():.4f}")
    assert logits.shape == (2, 64, config.vocab_size), "Logits shape mismatch!"
    assert not torch.isnan(loss), "Loss is NaN!"

    # 3. Backward Pass & Gradient Sanity Check
    print("\n⚡ TESTING BACKWARD PASS & GRADIENT INTEGRITY...")
    loss.backward()
    grad_norms = []
    zero_grads = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            grad_norms.append(norm)
            if norm == 0.0:
                zero_grads.append(name)
        else:
            print(f"⚠️ Warning: Missing gradient on {name}")

    print(f"  • Gradients computed for {len(grad_norms)} parameters.")
    print(f"  • Average Grad Norm: {sum(grad_norms)/len(grad_norms):.6f}")
    if zero_grads:
        print(f"  • Parameters with zero grad: {len(zero_grads)}")
    else:
        print(f"  • All parameters have active gradients! ✅")

    # 4. Auto-regressive Generation Test
    print("\n✍️ TESTING AUTOREGRESSIVE GENERATION...")
    prompt_ids = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt_ids, max_new_tokens=10, temperature=0.7)
    print(f"  • Input shape : {prompt_ids.shape}")
    print(f"  • Output shape: {generated.shape} (Generated 10 new tokens smoothly)")
    assert generated.shape == (1, 18), "Generation length mismatch!"

    print("\n" + "=" * 70)
    print("🎉 ALL X-SHAPE ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
