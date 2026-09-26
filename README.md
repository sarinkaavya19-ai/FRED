# FERD — Facial Emotion Recognition and Detection

> **Architecture Codename:** ST-ViT-GRU (Hybrid Spatio-Temporal Vision Transformer with Spatial-Attention Masking)
> **Repository:** `ferd-core`
> **Status:** ACTIVE — Pre-MVP

---

## 📌 Overview

**FERD** is an end-to-end trainable facial emotion recognition system designed to deliver real-time, cloud-grade accuracy on edge devices. Standard facial emotion recognition (FER) models often fail in real-world conditions due to lighting variations, face occlusions, lack of temporal memory, and excessive compute requirements.

FERD addresses these issues through a novel hybrid architecture (**ST-ViT-GRU**) that pairs spatial attention masking with a Mobile-ViT backbone and a temporal recurrent head. This enables on-device detection of both macro- and micro-expressions at **≥30 FPS** with robust performance under pose changes, partial occlusion, and low-light environments.

---

## ✨ Key Features & Highlights

* **Spatial-Attention Facial Masking (SAFM):** Dynamically down-weights feature noise caused by partial occlusion, low light, or extreme head poses before spatial encoding.
* **Mobile-ViT Backbone:** Uses a hybrid CNN-Transformer spatial encoder (`MobileViT-XS/S` teacher; `MobileViT-XXS` student) to capture local facial textures alongside global spatial context.
* **Temporal Memory (GRU / TCN):** Operates on a sliding-window ring buffer (default 16 frames / ~533ms) to capture continuous expression dynamics and involuntary micro-expressions (40ms–500ms).
* **Edge-First Optimization:** Features Knowledge Distillation (KD) and Post-Training Quantization (PTQ) / Quantization-Aware Training (QAT) to deliver an **INT8 TFLite/ONNX model binary under 12MB**.
* **Privacy & Compliance Ready:** Fully on-device inference with zero mandatory network frame egress, designed for automotive DMS (ISO 26262 readiness) and tele-health applications (HIPAA/GDPR compliance).

---

## 📊 Benchmark Targets vs. Baseline

| Metric | Commodity Baseline (Single-Frame CNN) | FERD (ST-ViT-GRU Target) |
| --- | --- | --- |
| **In-the-Wild Weighted-F1** | 61% – 68% | **≥ 88%** |
| **Occlusion / Low-Light Drop** | −15 to −30 pts | **≤ −5 pts** |
| **Edge Inference Latency** | 45–70ms (or 150–400ms cloud RTT) | **≤ 33ms (Fully on-device)** |
| **Target Hardware Frame Rate** | 14–22 FPS | **≥ 30 FPS** |
| **Deployable Binary Size** | 18MB – 40MB | **≤ 12MB (INT8 quantized)** |
| **Prediction Flicker Rate** | ~22% frame label switches | **≤ 9%** |
| **Micro-Expression Detection** | Not supported | **75% Onset Recall** |

---

## 🏗 System Architecture & Pipeline

```
┌───────────────────────┐
│  Camera Sensor Input  │
└───────────┬───────────┘
            │
┌───────────▼───────────────────────────┐
│ Face Detection & Landmark Alignment   │ (BlazeFace / MediaPipe, <2ms)
└───────────┬───────────────────────────┘
            │
┌───────────▼───────────────────────────┐
│ Spatial-Attention Facial Masking (SAFM)│ (Down-weights occluded/lit-poor regions)
└───────────┬───────────────────────────┘
            │
┌───────────▼───────────────────────────┐
│ Mobile-ViT Spatial Encoder            │ (256-dim embedding per frame)
└───────────┬───────────────────────────┘
            │
┌───────────▼───────────────────────────┐
│ Temporal Head (GRU / TCN)             │ (16-frame sliding window buffer)
└───────────┬───────────────────────────┘
            │
┌───────────▼───────────────────────────┐
│ Confidence-Scored Output API          │ (Emits 7 emotion classes, Valence-Arousal,
└───────────────────────────────────────┘  and Micro-expression onset flags)

```

---

## 🛠 Tech Stack

* **Frameworks:** PyTorch 2.x, PyTorch Lightning
* **Spatial Encoder:** MobileViT (`ferd-core/models/encoder/mobilevit.py`)
* **Temporal Head:** GRU / TCN (`ferd-core/models/temporal/gru_head.py`)
* **Attention Masking:** SAFM Custom Module (`ferd-core/models/safm/attention_mask.py`)
* **Face Mesh / Alignment:** MediaPipe (BlazeFace-based)
* **Model Compression & Quantization:** PTQ / QAT, Knowledge Distillation (`ferd-core/compression/`)
* **Export & Runtimes:** ONNX Runtime, TFLite, Core ML
* **Data & Experiment Tracking:** DVC (Data Version Control), Weights & Biases

---

## 📁 Repository Structure

```text
ferd-core/
├── ferd-core/
│   ├── preprocessing/      # Face detection & landmark alignment (MediaPipe/BlazeFace)
│   ├── models/             # Core architecture modules
│   │   ├── encoder/        # Mobile-ViT spatial encoder
│   │   ├── safm/           # Spatial-Attention Facial Masking (SAFM) module
│   │   └── temporal/       # GRU and TCN temporal heads
│   ├── training/           # PyTorch Lightning training routines & callbacks
│   ├── compression/        # Post-training quantization (PTQ) & distillation scripts
│   ├── export/             # ONNX, TFLite, and Core ML export pipelines
│   └── data/               # DVC dataset metadata & pipelines
├── docs/                   # Project documentation & PMD references
└── README.md

```

---

## 🚀 Key Non-Functional Requirements (NFRs)

* **End-to-End Latency:** $\le 33\text{ ms}$ (p99) / $\le 22\text{ ms}$ (p50) on target edge NPUs (Qualcomm QCS6490, Apple A16 Neural Engine).
* **Peak Memory Footprint:** $\le 180\text{ MB RAM}$ during inference execution.
* **Model Security:** Signed model binaries (Ed25519) verified prior to inference initialization.
* **Data Privacy:** On-device processing by default; zero raw frame network retention or transmission.

Prerequisites
# 1. Clone and navigate
git clone https://github.com/sarinkaavya19-ai/FRED.git
cd FRED/ferd-sprint

# 2. Verify environment (should already be set up)
python -c "import torch, timm, mediapipe, cv2; print('OK')"
Option 1: Pre-recorded Video Demo (Safest, Recommended)
# Run the inference pipeline on the test clip
python inference_demo.py
What you'll see: 60-frame test clip (15 frames each: Disgust, Fear, Surprise, Neutral) processed through full pipeline. Console shows per-frame predictions with confidence.
Option 2: Interactive Temporal Smoothing Demo (Webcam)
# Requires webcam; press keys during demo:
python demo_temporal.py
Controls during demo:
Key	Action
q	Quit
t	Toggle TEMPORAL (GRU) ↔ SINGLE-FRAME
h	Toggle confidence history plot
b	Toggle bounding box
s	Screenshot
What you'll see: Live webcam feed with:
- Face bounding box
- Top emotion + confidence %
- 7-class probability bars
- Confidence history plot (scrolling)
- Mode indicator (TEMPORAL vs SINGLE-FRAME)
Option 3: Basic Demo Harness (Webcam/Video/Image)
# Webcam
python demo.py

# Or video file
python demo.py --source video --video test_emotion_clip.mp4

# Or single image
python demo.py --source image --image path/to/face.jpg

# Force dummy mode (no model)
python demo.py --dummy
Option 4: Automated Rehearsal (Two Consecutive Runs)
# Runs full 6-minute demo script twice back-to-back
python rehearsal.py
What it does: Automated run through all 7 demo sections (PMD framing → Input → Single-frame → Temporal → Occlusion → No-face → Scope statement), timing each run.
Option 5: Run Edge Case Tests
# Tests 6 edge cases: blank, low light, overexposed, extreme pose, occlusion, baseline
python test_edge_cases.py
Expected: All 6 PASS (no crashes, graceful degradation)
Option 6: Verify Pipeline Components Individually
# 1. Face alignment only
python -c "
from preprocessing.face_align import FaceAligner
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
a = FaceAligner(running_mode='IMAGE')
r = a.align(frame)
print('Face align:', 'OK' if r.success else r.error)
a.close()
"

# 2. Encoder only
python -c "
import torch
from models.encoder.mobilevit import create_mobilevit_encoder
e = create_mobilevit_encoder('mobilevit_xs', pretrained=True, freeze=True)
x = torch.randn(1,3,224,224)
with torch.no_grad(): out = e(x)
print('Encoder:', out.shape)  # Should be (1, 256)
"

# 3. SAFM only
python -c "
import numpy as np
from models.safm.attention_mask import create_heuristic_safm
s = create_heuristic_safm()
img = np.random.rand(224,224,3).astype(np.float32)
lm = np.zeros((478,2)); lm[:,0]=112; lm[:,1]=112
mask, conf = s.generate_mask(img, lm)
print('SAFM mask:', mask.shape, 'range:', mask.min(), '-', mask.max())
"

# 4. Full pipeline (streaming)
python -c "
import torch
from pipeline import create_streaming_pipeline
p = create_streaming_pipeline()
p.eval()
x = torch.randn(1,3,224,224)
lm = torch.zeros(1,478,2)
vis = torch.ones(1,478)
with torch.no_grad(): r = p.step(x, lm, vis)
print('Pipeline:', r['probs'].shape, 'pred:', r['pred_class'].item())
"
Expected Outputs
Test	Success Indicator
inference_demo.py	Console shows 49/60 frames processed, predictions printed
demo_temporal.py	OpenCV window opens, shows face bbox + emotion + prob bars
rehearsal.py	Completes 2 runs, prints timing summary
test_edge_cases.py	All 6 tests PASS, no crashes
Troubleshooting
Issue	Fix
ModuleNotFoundError	Run from ferd-sprint/ directory or add to sys.path
Webcam not detected	Check cv2.VideoCapture(0) index; try 1, 2
Slow model loading	First run downloads MobileViT weights (~2MB); subsequent runs cached
Low FPS on CPU	Expected (~3-5 FPS); use pre-recorded demo for presentations
Unicode errors	Scripts use ASCII-safe output; ignore terminal encoding warnings
File Structure Reference
ferd-sprint/
├── demo.py              # Basic demo harness (webcam/video/image)
├── demo_temporal.py     # Temporal smoothing demo with toggles
├── inference_demo.py    # Pre-recorded clip inference
├── rehearsal.py         # Automated 2× demo rehearsal
├── test_edge_cases.py   # Edge case hardening tests
├── pipeline.py          # Core pipeline (batch + streaming)
├── preprocessing/face_align.py
├── models/
│   ├── encoder/mobilevit.py
│   ├── safm/attention_mask.py + region_confidence.py
│   ├── temporal/gru_head.py
│   └── heads/classification_head.py
├── data/dataset.py      # FER2013 DataLoader
└── data/datasets/fer2013/processed/
    ├── split.json       # Train/val/test split
    └── images/          # 35,887 images by class

