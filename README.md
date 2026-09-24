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
