# 🧬 Thai-0.1B X-Shape Transformer (24-Layer Architecture)

สถาปัตยกรรม **X-Shape (Hourglass / Variable-Width Transformer)** สำหรับโมเดลภาษาขนาดเล็ก (SLM) ขนาด ~105.54M พารามิเตอร์ ออกแบบให้มีความลึก **24 ชั้น (ลึกขึ้น 2 เท่าจากเดิม 12 ชั้น)** ภายใต้งบพารามิเตอร์ระดับ 0.1B เท่าเดิม

---

## 📐 สถาปัตยกรรม (Architecture Topology)

```text
[Input Tokens: Vocab 20,480]
              │
              ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Entrance Head (4 Layers @ Dim 768) │ ──┐
│  - FFN: 2,048 SwiGLU                         │   │
│  - GQA: 12 Query Heads, 4 KV Heads           │   │
└──────────────────────────────────────────────┘   │  [X-Cross Skip Connection]
              │                                    │  (ส่ง Low-level Morphology ตรงไปท้าย)
              ▼                                    │
    [Down-Bridge: 768 ➔ 480]                       │
              │                                    │
              ▼                                    │
┌──────────────────────────────────────────────┐   │
│  Stage 2: Deep Core Bottleneck (16 Layers!)  │   │
│  - Dim: 480 | FFN: 1,280 SwiGLU              │   │
│  - GQA: 8 Query Heads, 2 KV Heads            │   │
│  - 16 Deep Iterative Reasoning Layers        │   │
└──────────────────────────────────────────────┘   │
              │                                    │
              ▼                                    │
     [Up-Bridge: 480 ➔ 768]                        │
              │                                    │
              ▼                                    │
     [Residual Addition] ◄─────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Exit Decoder (4 Layers @ Dim 768)  │
│  - FFN: 2,048 SwiGLU                         │
│  - GQA: 12 Query Heads, 4 KV Heads           │
└──────────────────────────────────────────────┘
              │
              ▼
   [Final RMSNorm & LM Head (Tied)]
```

---

## 📊 ตารางเปรียบเทียบ: Standard V2 vs X-Shape V3

| คุณสมบัติ | Standard Transformer (V2) | X-Shape Transformer (V3) |
| :--- | :--- | :--- |
| **จำนวนชั้นรวม (Total Depth)** | **12 ชั้น** | **24 ชั้น (ลึกขึ้น 2 เท่า!)** |
| • Stage 1 (Head) | 12 ชั้น @ Dim 768 | 4 ชั้น @ Dim 768 (25.17M) |
| • Stage 2 (Core) | - | **16 ชั้น @ Dim 480 (FFN 1,344)** (40.20M) |
| • Stage 3 (Tail) | - | 4 ชั้น @ Dim 768 (25.17M) |
| • Bridges & Cross-Skip | - | **Learnable Gated Skip + Bridges** (1.33M) |
| • Embeddings (Tied) | 15.73M | 15.73M |
| **พารามิเตอร์รวมสุทธิ** | **107.96M (0.108B)** | **107.60M (0.108B)** |
| **Vocab Size** | 20,480 (Thai-20k) | 20,480 (Thai-20k) |
| **ความเร็วในการรัน (Inference)** | มาตรฐาน | **เร็วกว่าเดิม ~35%** (เพราะชั้นกลางแคบ) |

---

## 🛠️ ไฟล์ทั้งหมดในโฟลเดอร์นี้

* [`config_x_shape.py`](file:///Users/phonsirithabunsri/Desktop/0.1B/X-Shape/config_x_shape.py) : คอนฟิกโครงสร้างโมเดล 24 ชั้น
* [`modeling_x_shape.py`](file:///Users/phonsirithabunsri/Desktop/0.1B/X-Shape/modeling_x_shape.py) : โค้ด PyTorch เต็มรูปแบบ (RoPE, GQA, SDPA FlashAttention, SwiGLU, Bridges, Cross-Skip)
* [`test_x_shape_arch.py`](file:///Users/phonsirithabunsri/Desktop/0.1B/X-Shape/test_x_shape_arch.py) : สคริปต์รันทดสอบ Sanity Check (Forward, Backward, Gradient Norm, Auto-regressive Generation)
* [`train_x_shape_modal.py`](file:///Users/phonsirithabunsri/Desktop/0.1B/X-Shape/train_x_shape_modal.py) : สคริปต์เทรนบน Modal Cloud (L4 GPU 24GB + Streaming Balanced Dataset)

---

## 🚀 วิธีรัน

### 1. ทดสอบโครงสร้างสถาปัตยกรรมในเครื่อง Mac:
```bash
python3 test_x_shape_arch.py
```

### 2. สั่งรัน Pretraining ขึ้นคลาวด์ Modal:
```bash
modal run --detach train_x_shape_modal.py::train_x_shape
```
