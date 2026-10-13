"""
Modal Pretraining Script for X-Shape 24-Layer Transformer (0.1B SLM)
====================================================================
Architecture:
  - Total Layers: 24 (4 Entrance + 16 Core Bottleneck + 4 Exit Decoder)
  - Hidden sizes: 768 -> 480 -> 768 with Cross-Skip X-Residual
  - Total Parameters: 105.54M (0.105B)
  - Tokenizer: Jommarn/thai-0.1b-tokenizer-20k
Data Mixture:
  - 30% Jommarn/thai-semi-pretrain-qwen
  - 35% wikimedia/wikipedia (Thai)
  - 20% HuggingFaceFW/fineweb-2 (tha_Thai)
  - 15% HuggingFaceFW/fineweb-edu (sample-10BT)
"""

import os
import sys
import time
import math
import random
import modal

app = modal.App("thai-0.1b-x-shape-pretrain")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "huggingface_hub>=0.22.0",
    )
    .add_local_file("config_x_shape.py", "/root/config_x_shape.py")
    .add_local_file("modeling_x_shape.py", "/root/modeling_x_shape.py")
)

volume = modal.Volume.from_name("x-shape-checkpoints-volume", create_if_missing=True)

def get_local_hf_token():
    try:
        from huggingface_hub import get_token
        return get_token() or ""
    except Exception:
        return ""

LOCAL_HF_TOKEN = get_local_hf_token()

TRAIN_CONFIG = {
    "total_steps": 10000,
    "batch_size": 8,
    "grad_accum": 2,
    "max_seq_len": 2048,
    "learning_rate": 3.0e-4,
    "min_learning_rate": 3.0e-5,
    "warmup_steps": 250,
    "save_every": 250,
    "eval_every": 250,
    "hf_repo_id": "Jommarn/thai-0.1b-x-shape",
}

@app.function(
    gpu="L4",
    image=image,
    volumes={"/root/checkpoints_x_shape": volume},
    timeout=86400,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": LOCAL_HF_TOKEN})] if LOCAL_HF_TOKEN else [],
)
def train_x_shape():
    import sys
    import torch
    from torch.utils.data import IterableDataset, DataLoader
    from transformers import PreTrainedTokenizerFast
    from datasets import load_dataset
    from huggingface_hub import HfApi, login

    sys.path.insert(0, "/root")
    from config_x_shape import XShapeConfig
    from modeling_x_shape import XShapeForCausalLM

    print("=" * 70)
    print("🚀 INITIALIZING X-SHAPE 24-LAYER PRETRAINING ON MODAL (NVIDIA L4)")
    print("=" * 70)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
        print("✅ Authenticated with Hugging Face Hub.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Tokenizer
    tok_id = "Jommarn/thai-0.1b-tokenizer-20k"
    print(f"📦 Loading Tokenizer: {tok_id}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    # 2. Build Model & Auto-Resume
    config = XShapeConfig()
    model = XShapeForCausalLM(config)
    start_step = 0

    if os.path.exists("/root/checkpoints_x_shape"):
        saved_steps = []
        for d in os.listdir("/root/checkpoints_x_shape"):
            if d.startswith("step_"):
                try:
                    s = int(d.split("_")[1])
                    p = f"/root/checkpoints_x_shape/{d}/model_checkpoint.pt"
                    if os.path.exists(p):
                        saved_steps.append((s, p))
                except Exception:
                    pass
        if saved_steps:
            saved_steps.sort(key=lambda x: x[0], reverse=True)
            latest_step, latest_ckpt = saved_steps[0]
            print(f"🔄 Found existing checkpoint at Step {latest_step}! Resuming from {latest_ckpt}...")
            ckpt_data = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
            start_step = latest_step
            print(f"✅ Successfully resumed weights from Step {start_step}!")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model Architecture: X-Shape 24-Layer Transformer")
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Upload tokenizer to HF repo once at start
    if hf_token and start_step == 0:
        try:
            tokenizer.push_to_hub(TRAIN_CONFIG["hf_repo_id"])
            print(f"📦 Pushed Tokenizer to https://huggingface.co/{TRAIN_CONFIG['hf_repo_id']}")
        except Exception as e:
            print(f"Notice pushing tokenizer: {e}")

    # 3. Data Loading
    print("📚 Preparing 4-Way Balanced Streaming Dataset...")
    qwen_clean = [x["text"] for x in load_dataset("Jommarn/thai-semi-pretrain-qwen", split="train_clean")]
    qwen_think = [x["text"] for x in load_dataset("Jommarn/thai-semi-pretrain-qwen", split="train_think")]
    print(f"✅ Qwen Clean cached: {len(qwen_clean):,} docs | Qwen Think cached: {len(qwen_think):,} docs")

    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.th", split="train", streaming=True)
    ds_fw2 = load_dataset("HuggingFaceFW/fineweb-2", "tha_Thai", split="train", streaming=True)
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    iter_wiki = iter(ds_wiki)
    iter_fw2 = iter(ds_fw2)
    iter_edu = iter(ds_edu)

    class MultiStreamPackedDataset(IterableDataset):
        def __init__(self, tokenizer, max_seq_len):
            self.tokenizer = tokenizer
            self.max_seq_len = max_seq_len

        def __iter__(self):
            nonlocal iter_wiki, iter_fw2, iter_edu
            token_buffer = []

            while True:
                r = random.random()
                text = None
                if r < 0.20:
                    text = random.choice(qwen_clean)
                elif r < 0.30:
                    text = random.choice(qwen_think)
                elif r < 0.65:
                    try:
                        sample = next(iter_wiki)
                        text = sample.get("text", "")
                    except StopIteration:
                        iter_wiki = iter(ds_wiki)
                        text = next(iter_wiki).get("text", "")
                elif r < 0.85:
                    try:
                        sample = next(iter_fw2)
                        text = sample.get("text", "")
                    except StopIteration:
                        iter_fw2 = iter(ds_fw2)
                        text = next(iter_fw2).get("text", "")
                else:
                    try:
                        sample = next(iter_edu)
                        text = sample.get("text", "")
                    except StopIteration:
                        iter_edu = iter(ds_edu)
                        text = next(iter_edu).get("text", "")

                if not text or len(text.strip()) < 10:
                    continue

                tokens = self.tokenizer.encode(text) + [eos_id]
                token_buffer.extend(tokens)

                while len(token_buffer) >= self.max_seq_len:
                    chunk = token_buffer[:self.max_seq_len]
                    token_buffer = token_buffer[self.max_seq_len:]
                    t = torch.tensor(chunk, dtype=torch.long)
                    yield t, t

    train_ds = MultiStreamPackedDataset(tokenizer, TRAIN_CONFIG["max_seq_len"])
    loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], num_workers=0)
    data_iter = iter(loader)

    # 4. Optimizer & LR Schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    def get_lr(step):
        if step < TRAIN_CONFIG["warmup_steps"]:
            return TRAIN_CONFIG["learning_rate"] * (step / TRAIN_CONFIG["warmup_steps"])
        progress = (step - TRAIN_CONFIG["warmup_steps"]) / (TRAIN_CONFIG["total_steps"] - TRAIN_CONFIG["warmup_steps"])
        return TRAIN_CONFIG["min_learning_rate"] + 0.5 * (TRAIN_CONFIG["learning_rate"] - TRAIN_CONFIG["min_learning_rate"]) * (1.0 + math.cos(math.pi * progress))

    # 5. Training Loop
    total_steps = TRAIN_CONFIG["total_steps"]
    grad_accum = TRAIN_CONFIG["grad_accum"]
    print(f"🔥 Starting Training from Step {start_step + 1} to {total_steps:,} Steps with bfloat16 mixed precision...")

    model.train()
    start_time = time.time()
    tokens_per_step = TRAIN_CONFIG["batch_size"] * grad_accum * TRAIN_CONFIG["max_seq_len"]

    eval_prompts = [
        "กรุงเทพมหานคร เป็นเมืองหลวงของประเทศไทย โดยมี",
        "ดวงอาทิตย์ ขึ้นทางทิศ",
        "วันสงกรานต์ ตรงกับเดือน",
        "ประเทศไทย มีเมืองหลวงชื่อว่า",
        "ปัญญาประดิษฐ์ หรือ AI คือ",
        "<|im_start|>user\nสวัสดีครับ คุณคือใคร<|im_end|>\n<|im_start|>assistant\n",
    ]

    for step in range(start_step + 1, total_steps + 1):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, labels=y)
                loss = out["loss"] / grad_accum

            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5 == 0 or step == start_step + 1:
            elapsed = time.time() - start_time
            tok_speed = (tokens_per_step * 5) / max(elapsed, 1e-4) if step > start_step + 1 else tokens_per_step / max(elapsed, 1e-4)
            start_time = time.time()
            vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"🔥 Step {step:5d}/{total_steps} | Loss: {accum_loss:6.4f} | LR: {lr:.2e} | VRAM: {vram_gb:.2f}GB / 24GB | Speed: {tok_speed:,.0f} tok/s")

        # Save & Live Generation Evaluation
        if step % TRAIN_CONFIG["save_every"] == 0 or step == total_steps:
            model.eval()
            print("\n" + "=" * 60)
            print(f"🧪 [EVALUATION AT STEP {step}] Live Text Generation:")
            print("=" * 60)
            for ep in eval_prompts:
                inp = tokenizer.encode(ep, return_tensors="pt").to(device)
                curr = inp.clone()
                with torch.no_grad():
                    for _ in range(25):
                        out_ev = model(curr)
                        nxt = out_ev["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                        curr = torch.cat([curr, nxt], dim=-1)
                        if nxt.item() == eos_id:
                            break
                gen_text = tokenizer.decode(curr[0].tolist(), skip_special_tokens=False)[len(ep):]
                print(f"[Prompt]: \"{ep.strip()}\"")
                print(f"[Generated]: '{gen_text.strip()}'")
                print("-" * 50)
            model.train()

            save_dir = f"/root/checkpoints_x_shape/step_{step}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "model_checkpoint.pt")
            torch.save({
                "step": step,
                "loss": accum_loss,
                "config": config,
                "model_state_dict": model.state_dict(),
            }, save_path)
            volume.commit()
            print(f"💾 [CHECKPOINT] Saved X-Shape Checkpoint at Step {step} to {save_path}")

            if hf_token:
                try:
                    HfApi().upload_file(
                        path_or_fileobj=save_path,
                        path_in_repo="model_checkpoint.pt",
                        repo_id=TRAIN_CONFIG["hf_repo_id"],
                        commit_message=f"Upload Thai-0.1B-X-Shape step {step} (Loss {accum_loss:.4f})",
                    )
                    print(f"✅ Live on Hugging Face: https://huggingface.co/{TRAIN_CONFIG['hf_repo_id']}")
                except Exception as e:
                    print(f"HF upload notice: {e}")

    print("\n🎉 X-SHAPE PRETRAINING COMPLETED SUCCESSFULLY!")

@app.local_entrypoint()
def main():
    print("🚀 Triggering X-Shape 24-Layer Pretraining on Modal...")
    train_x_shape.remote()
