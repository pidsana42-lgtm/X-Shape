"""
Local Evaluation Script for Thai-0.1B-X-Shape (24-Layer Transformer)
Downloads the latest checkpoint from Hugging Face and tests generation on Apple Silicon (MPS / CPU).
"""

import sys
import os
import time
import torch
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

from config_x_shape import XShapeConfig
from modeling_x_shape import XShapeForCausalLM

def main():
    print("=" * 70)
    print("🔬 EVALUATING THAI-0.1B-X-SHAPE (24-LAYER TRANSFORMER)")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Inference Hardware: {device}")

    # 1. Load Tokenizer
    repo_id = "Jommarn/thai-0.1b-x-shape"
    print(f"📦 Loading Tokenizer from {repo_id}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(repo_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    print(f"   Vocab Size: {tokenizer.vocab_size:,}")

    # 2. Download latest checkpoint from HF Hub
    print(f"📥 Downloading latest model_checkpoint.pt from {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model_checkpoint.pt")
    print(f"   Checkpoint cached at: {ckpt_path}")

    # 3. Load Model
    print("🧠 Initializing 24-Layer X-Shape Architecture...")
    config = XShapeConfig()
    model = XShapeForCausalLM(config)

    print("🔄 Loading checkpoint weights...")
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ckpt_data.get("step", "Unknown")
    loss = ckpt_data.get("loss", "Unknown")
    print(f"   ★ Checkpoint Step : {step}")
    print(f"   ★ Checkpoint Loss : {loss:.4f}" if isinstance(loss, float) else f"   ★ Checkpoint Loss : {loss}")

    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(device)
    model.eval()
    print("✅ Model loaded and ready for inference!\n")

    # 4. Test Prompts Suite
    test_prompts = [
        # ข้อเท็จจริงภูมิศาสตร์และการเมือง
        "กรุงเทพมหานคร เป็นเมืองหลวงของประเทศไทย โดยมี",
        "ประเทศไทย มีเมืองหลวงชื่อว่า",
        # วิทยาศาสตร์และธรรมชาติ
        "ดวงอาทิตย์ ขึ้นทางทิศ",
        "สิ่งมีชีวิต ต้องการน้ำและ",
        # วัฒนธรรมและประเพณี
        "วันสงกรานต์ ตรงกับเดือน",
        "ต้มยำกุ้ง เป็นอาหารไทยที่มีรสชาติ",
        # เทคโนโลยี
        "ปัญญาประดิษฐ์ หรือ AI คือ",
        "คอมพิวเตอร์ เป็นอุปกรณ์อิเล็กทรอนิกส์ที่",
        # สิ่งมีชีวิตและสัตว์เลี้ยง
        "แมว เป็นสัตว์เลี้ยงที่มีลักษณะ",
        # ภาษา
        "ภาษาไทย เป็นภาษาตระกูล",
    ]

    print("=" * 70)
    print("📝 GENERATION TEST (Greedy Search + Sampling)")
    print("=" * 70)

    for i, prompt in enumerate(test_prompts, 1):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        # Greedy generation
        start_t = time.time()
        with torch.no_grad():
            curr = input_ids.clone()
            for _ in range(40):
                out = model(curr)
                logits = out["logits"][:, -1, :]
                nxt = logits.argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=-1)
                if nxt.item() == eos_id:
                    break

        elapsed = time.time() - start_t
        gen_tokens = curr.shape[1] - prompt_len
        tok_speed = gen_tokens / max(elapsed, 1e-4)

        gen_text = tokenizer.decode(curr[0].tolist(), skip_special_tokens=True)
        completion = gen_text[len(prompt):].strip()

        print(f"\n[{i}/{len(test_prompts)}] Prompt: \"{prompt}\"")
        print(f"➡️ Completion: {completion}")
        print(f"⚡ Speed: {tok_speed:.1f} tok/s ({gen_tokens} tokens in {elapsed:.2f}s)")
        print("-" * 60)

    print("\n🎉 Evaluation Complete!")

if __name__ == "__main__":
    main()
