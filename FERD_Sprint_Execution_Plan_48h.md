# FERD — 48-Hour Sprint Execution Plan
## ST-ViT-GRU MVP Slice: Tactical Delivery Runbook

| Field | Value |
|---|---|
| Parent Document | FERD Project Master Document (PMD) v1.0 |
| Document Type | Sprint Execution Plan — Tactical, NOT a scope replacement |
| Sprint Window | 48 hours, fixed (Day 1 00:00 → Day 2 24:00, relative sprint clock) |
| Team | 2 engineers — Eng-A (ML/CV lead), Eng-B (pipeline/infra + demo lead) |
| Compute | 1x GPU dev workstation or equivalent single cloud GPU instance (RTX 4090-class / A100 fallback). No training cluster, no edge/NPU hardware. |
| Sprint Goal | Internal proof-of-concept demo validating the ST-ViT-GRU pipeline shape end-to-end |
| Assumption Basis | Team size, compute tier, and sprint goal were not specified in the brief; locked as stated above for planning purposes. Revise Section 2 timings if compute tier differs. |

---

## 1. Scope Lock & Cut List

### 1.1 IN for this sprint

| Item | Sprint Form |
|---|---|
| Mobile-ViT spatial encoder | Pretrained MobileViT-XS (ImageNet weights via `timm`), **not trained from scratch** |
| Spatial-Attention Facial Masking (SAFM) | Simplified, **non-learned** heuristic version: landmark-visibility + local-contrast scoring feeds a soft mask directly (Section 6.1 of PMD's formulation), skipping the trainable gating network |
| Temporal head | GRU head, single configuration only (TCN alternative dropped) — 16-frame sliding window as specified in PMD Section 4.1 |
| Face detection/alignment | MediaPipe Face Mesh, used as-is (already a mature off-the-shelf component per PMD Section 7) |
| Classification head | 7-class categorical output only (valence-arousal regression head dropped) |
| Training data | A small, pre-licensed public subset (FER2013 or a curated AffectNet sample, whichever is accessible without a new licensing cycle) — fine-tuning the classification + temporal head only, encoder frozen or lightly fine-tuned |
| Inference target | CPU/GPU dev machine, pre-recorded video file **and** live webcam if time allows — **not** an edge/NPU target |
| Output | Console/UI overlay showing 7-class distribution + confidence, updated per frame, temporally smoothed |

### 1.2 Explicitly DEFERRED (and why it's safe to cut)

| Cut Item | Why Safe to Defer |
|---|---|
| Full teacher training on AffectNet-wild + DFEW combined | Requires days of GPU-hours and full dataset licensing/ingestion (PMD Section 5.2 budgets 9 months / 8x A100-class GPU-hours for this) — categorically incompatible with 48 hours |
| Trainable SAFM gating network | The learned version needs its own training loop and labeled occlusion data; the heuristic version demonstrates the *architectural concept* (down-weighting unreliable regions) without the training cost |
| Knowledge distillation (teacher → student) | Distillation requires a fully converged teacher first — nothing to distill from in this window |
| PTQ/QAT quantization | Meaningless without a finalized, converged model to quantize |
| ONNX/TFLite/Core ML export | Edge export toolchain is a Phase 1 (Months 1–4) deliverable in the PMD, not a 2-day one |
| Edge/NPU deployment & latency validation | No edge hardware in scope; sub-33ms edge NFR (PMD Section 6.4) is untestable without target silicon |
| Micro-expression onset/offset head | Requires CASME II/SAMM data and its own auxiliary training — a Phase 2 PMD deliverable |
| Valence-arousal regression head | Secondary output; adds training complexity for no demo value in 48 hours |
| Multi-face tracking / session state machine | Session-manager complexity (PMD Section 8, FERD-F07) is unnecessary for a single-subject demo |
| Model signing, MLflow registry, CI/CD gates | Production-readiness infrastructure, irrelevant to a local demo build |
| Privacy/compliance review, ISO 26262 docs | Governance artifacts scoped to Phase 2 design-partner pilots, not a 2-day PoC |
| Telemetry/observability stack | No fleet to observe |

### 1.3 Definition of Done (single sentence)

> **By Hour 48, a fine-tuned ST-ViT-GRU pipeline (MobileViT-XS encoder + heuristic SAFM + GRU temporal head) running locally on the dev machine ingests a pre-recorded or live webcam video and outputs a real-time, temporally-smoothed 7-class emotion classification with confidence scores, displayed on screen — demonstrating the full architectural shape of FERD end-to-end, live, on camera.**

---

## 2. Sprint Timeline Overview

| Day | Time Block | Sprint Goal | Owner | Deliverable | Go/No-Go Checkpoint |
|---|---|---|---|---|---|
| 1 | H0–H2 | Environment, repo skeleton, data acquisition | Eng-A + Eng-B | Working dev environment, dataset subset pulled | **GO/NO-GO 1** (H2): Data + env ready or fallback dataset triggered |
| 1 | H2–H5 | Face detection + alignment pipeline wired | Eng-B | `face_align.py` producing aligned crops from video/webcam | Pipeline runs on sample clip without crashing |
| 1 | H5–H8 | MobileViT-XS encoder integrated (pretrained) | Eng-A | Encoder producing per-frame embeddings | Embedding shape/sanity check passes |
| 1 | H8–H11 | Heuristic SAFM module | Eng-A | Soft attention mask applied pre-encoder | Visual mask sanity check on occluded test frame |
| 1 | H11–H14 | GRU temporal head + classification head wired (untrained) | Eng-B | End-to-end forward pass, random-init temporal head | **GO/NO-GO 2** (H14): Full pipeline runs start-to-finish, even if untrained |
| 1 | H14–H18 | Training data prep + label pipeline | Eng-A | Train/val split ready, dataloader validated | Dataloader yields correct batch shapes |
| 1 | H18–H22 | First fine-tuning run kicked off | Eng-A | Loss curve visible, decreasing | **GO/NO-GO 3** (H22): Loss non-NaN and trending down, or fallback triggered |
| 1 | H22–H24 | Overnight training run + demo harness scaffolding | Eng-A (training) / Eng-B (harness) | Training running unattended; demo UI skeleton exists | — |
| 2 | H24–H28 | Check training result; resume/adjust | Eng-A | Checkpoint evaluated on val subset | **GO/NO-GO 4** (H28): Val accuracy above chance or checkpoint/pretrained fallback triggered |
| 2 | H28–H32 | Inference pipeline wired to trained checkpoint | Eng-A + Eng-B | Live classification output on recorded clip | Output visibly correlates with ground-truth expression |
| 2 | H32–H36 | Temporal smoothing tuning + confidence display | Eng-B | Smoothed, readable on-screen overlay | Flicker visibly reduced vs. raw per-frame output |
| 2 | H36–H40 | Webcam live-inference integration | Eng-B | Live webcam demo runs at usable FPS | **GO/NO-GO 5** (H40): Live webcam path works, or pre-recorded-only fallback locked in |
| 2 | H40–H44 | Bug bash, edge-case hardening, demo polish | Eng-A + Eng-B | No-face / low-light behavior doesn't crash | Demo survives 3 consecutive dry runs |
| 2 | H44–H46 | Final demo rehearsal | Eng-A + Eng-B | Rehearsed run-through, timing confirmed | Full script rehearsed twice, no crashes |
| 2 | H46–H48 | **Demo Freeze** — no new code | Eng-A + Eng-B | Frozen, demo-ready build | **FINAL GO** (H46): Freeze declared |

---

## 3. Day 1 — Detailed Task Breakdown

### Block 1.1 (H0–H2): Environment & Data Acquisition
**Tasks**
- Provision GPU dev environment; clone/scaffold `ferd-sprint` repo (trimmed fork of `ferd-core` structure — encoder/, safm/, temporal/, preprocessing/ only)
- Install `torch`, `timm` (for pretrained MobileViT), `mediapipe`, `opencv-python`, `pytorch-lightning` (optional, may skip for speed)
- Identify and pull the training data subset: prefer FER2013 (small, no licensing friction, instantly downloadable) over a full AffectNet-wild pull (large, may require licensing lead time per PMD Section 5.2)
- Confirm webcam/video I/O works on the dev machine

**Acceptance Criteria**
- `python -c "import torch, timm, mediapipe, cv2"` succeeds with no errors
- Dataset subset (≥3,000 labeled images across the 7 classes) is on local disk
- A test video file plays back and is readable via OpenCV

**Blocking Dependencies:** None (first block)

**Fallback if overrun:** If AffectNet access is blocked or slow, switch to FER2013 immediately — do not wait past H1:30 for a licensing/download response. This decision is pre-approved; no need to re-confirm mid-sprint.

---

### Block 1.2 (H2–H5): Face Detection & Alignment
**Tasks**
- Wire MediaPipe Face Mesh into `preprocessing/face_align.py`, reusing the PMD's chosen tool (Section 7) as-is
- Output: aligned, cropped face tensor per frame at fixed resolution (e.g., 224x224)
- Validate against a sample webcam feed and a sample pre-recorded clip

**Acceptance Criteria**
- Given any input frame with a visible face, the module returns a correctly cropped/aligned tensor within <10ms on CPU
- Given a frame with no face, the module returns a well-defined "no face" signal (not a crash)

**Blocking Dependencies:** Block 1.1 environment must be complete

**Fallback if overrun:** If MediaPipe integration stalls past H4:30, drop to a simpler face-crop-only approach (OpenCV Haar cascade + fixed-margin crop, no landmark alignment) to unblock the encoder work in Block 1.3. Alignment quality is not demo-critical; pipeline completeness is.

---

### Block 1.3 (H5–H8): MobileViT-XS Encoder Integration
**Tasks**
- Load pretrained MobileViT-XS via `timm` (ImageNet weights)
- Strip/replace the classification head; expose the penultimate embedding layer
- Run a forward pass on aligned face crops from Block 1.2, confirm embedding dimensionality matches what the GRU head will expect

**Acceptance Criteria**
- Encoder forward pass runs on both CPU and GPU without shape errors
- Per-frame embedding vector is deterministic and stable across repeated runs on the same input

**Blocking Dependencies:** Block 1.2 output tensors

**Fallback if overrun:** If MobileViT-XS via `timm` has integration friction, fall back to MobileViT-XXS (smaller, faster to iterate) or, as a last resort, a standard EfficientNet-B0 pretrained backbone — architecturally off-brief but keeps the sprint moving. Document this substitution explicitly in Section 7 (Handoff Notes) if used.

---

### Block 1.4 (H8–H11): Heuristic SAFM Module
**Tasks**
- Implement the non-learned SAFM: compute per-region visibility from MediaPipe landmark confidence + local contrast (brightness variance) per facial region (eyes, brows, mouth, jaw)
- Convert region scores into a soft spatial mask applied to the input crop (or to intermediate feature map) before/during encoding
- Visual sanity check: manually occlude part of a test face (e.g., hand over mouth) and confirm the mask down-weights that region

**Acceptance Criteria**
- Mask output is a smooth, non-binary weighting (not a hard on/off mask) — this is the architectural point being demonstrated
- On an occluded test frame, the occluded region's mask weight is visibly lower than an unoccluded region's

**Blocking Dependencies:** Block 1.2 (landmark output) and Block 1.3 (encoder must accept masked input)

**Fallback if overrun:** If the learned-vs-heuristic mask integration proves fiddly, apply the mask as a pre-multiplication on the input image directly (crude but functional) rather than injecting it into intermediate encoder features. This is a known simplification — flag it in Section 7.

---

### Block 1.5 (H11–H14): Temporal Head Wiring (Untrained) — GO/NO-GO 2
**Tasks**
- Implement GRU head (`temporal/gru_head.py`): consumes a 16-frame sliding window of per-frame embeddings, outputs a hidden state
- Implement classification head: 7-class softmax on top of the GRU's final hidden state
- Wire full pipeline: video/webcam → face align → SAFM → encoder → embedding buffer → GRU → classification head
- Run one full end-to-end forward pass, even with random-initialized weights, purely to confirm the pipeline doesn't break

**Acceptance Criteria**
- Full pipeline runs start-to-finish on a 16-frame clip without shape/runtime errors
- Output is a valid 7-class probability distribution (sums to 1), even if meaningless pre-training

**GO/NO-GO 2 Decision:** If the full pipeline does not run end-to-end by H14, **stop feature work** and spend the remainder of Block 1.5 exclusively on getting a forward pass working, even in a degraded/simplified form (e.g., skip SAFM, feed encoder output directly to GRU). A working (if untrained) pipeline by H14 is the single hardest gate in the sprint — everything downstream depends on it.

**Blocking Dependencies:** Blocks 1.2, 1.3, 1.4 all complete

**Fallback if overrun:** Simplify to single-frame classification (drop the GRU temporally) as an emergency fallback architecture, and revisit adding the temporal head in Block 2.1 if time recovers. This is the single biggest scope-cut lever available if Day 1 is behind schedule.

---

### Block 1.6 (H14–H18): Training Data Prep
**Tasks**
- Build the PyTorch `Dataset`/`DataLoader` for the fine-tuning set (images, or short clips if using a video-based dataset)
- Train/val split (e.g., 85/15)
- If using a static-image dataset (FER2013) rather than video: synthesize short "windows" by repeating/lightly augmenting the same frame across the 16-frame window, since true temporal fine-tuning data is out of reach in this window — this is a known, documented simplification (see Section 7)

**Acceptance Criteria**
- `DataLoader` yields correctly shaped batches matching the pipeline's expected input
- Class distribution across train/val is logged and roughly balanced (or class weights computed if not)

**Blocking Dependencies:** Block 1.1 (data on disk), Block 1.5 (pipeline shape confirmed)

**Fallback if overrun:** If dataloader/label wrangling eats time, hard-cap the dataset to a smaller balanced subset (e.g., 500 images/class) rather than debugging a larger, messier pull. Smaller and clean beats larger and buggy at this compute/time budget.

---

### Block 1.7 (H18–H22): First Fine-Tuning Run — GO/NO-GO 3
**Tasks**
- Freeze MobileViT-XS encoder weights (or unfreeze last 1–2 blocks only); train classification head + GRU head only
- Kick off training with a conservative learning rate, log loss every N steps
- Monitor for NaN loss, non-decreasing loss, or GPU OOM

**Acceptance Criteria**
- Loss is non-NaN and visibly decreasing within the first 30 minutes of training
- No GPU OOM at the batch size chosen

**GO/NO-GO 3 Decision (H22):** If loss has not started decreasing by H21, **do not keep debugging the training run into the night unattended** — stop, reduce learning rate or batch size once, restart, and let it run overnight regardless of the first hour's trajectory. A partially-converged model by H24 beats a perfectly-tuned one that never finished.

**Blocking Dependencies:** Block 1.6 dataloader validated

**Fallback if overrun:** If training instability persists after one restart, fall back to training the classification head only (freeze GRU at random init, or drop GRU into a simple temporal-averaging pooling instead of a trained recurrent unit) — this still demonstrates the architecture's shape, just with a simpler temporal aggregation.

---

### Block 1.8 (H22–H24): Overnight Run + Demo Harness Scaffold
**Tasks**
- Eng-A: let training run unattended overnight (checkpoint every N epochs; do not babysit)
- Eng-B: build the demo harness skeleton in parallel — video/webcam capture loop, on-screen overlay rendering (bounding box + text overlay for emotion label/confidence), independent of whether the model is trained yet (use dummy/random outputs to build the UI)

**Acceptance Criteria**
- Training job is running and checkpointing without supervision
- Demo harness renders a bounding box and dummy classification text on a live video feed

**Blocking Dependencies:** Block 1.7 (training launched), Block 1.5 (pipeline interface defined)

**Fallback if overrun:** None needed — this block is explicitly designed to run unattended/in parallel. If the demo harness isn't finished by H24, it continues into Block 2.1 without disrupting the training run.

---

## 4. Day 2 — Detailed Task Breakdown

### Block 2.1 (H24–H28): Checkpoint Evaluation — GO/NO-GO 4
**Tasks**
- Load the best overnight checkpoint; evaluate on the held-out val split
- Compute basic accuracy/weighted-F1 (not the full PMD KR1.1 benchmark — just a sanity metric)
- Spot-check qualitative predictions on a few val images by eye

**Acceptance Criteria**
- Val accuracy is meaningfully above chance level (>1/7 ≈ 14% for 7-class) — this is a low bar deliberately, since full convergence is not the goal
- Predictions are not collapsed to a single class for all inputs

**GO/NO-GO 4 Decision (H28):** 
- If val accuracy is reasonable (materially above chance, predictions vary sensibly) → proceed to Block 2.2 with this checkpoint.
- If accuracy is at/near chance or collapsed → **do not retrain from scratch.** Fall back to one of: (a) unfreeze more encoder layers and fine-tune for 2 more hours, capped hard at H30, or (b) demo with a partially-trained checkpoint and be transparent in the demo narration that this is an early-stage fine-tune, not a converged model. Either fallback is acceptable; retraining from scratch is not, given the remaining time budget.

**Blocking Dependencies:** Block 1.7/1.8 training run complete

**Fallback if overrun:** Same as above — checkpoint from a partial run is always preferable to burning remaining hours chasing a fully converged model that the sprint budget was never going to support (PMD's own KR1.1 target assumes a 9-month program, not 48 hours).

---

### Block 2.2 (H28–H32): Inference Pipeline on Trained Checkpoint
**Tasks**
- Swap the trained checkpoint into the full pipeline (replacing the random-init weights from Block 1.5)
- Run inference on a pre-recorded test clip with known/expected emotional content
- Confirm output classifications track the clip's content directionally (doesn't need to be perfect — needs to be visibly non-random)

**Acceptance Criteria**
- Running the pipeline on a labeled test clip produces the correct class as the top-1 or top-2 prediction for at least a majority of clearly-expressive segments

**Blocking Dependencies:** Block 2.1 checkpoint approved

**Fallback if overrun:** If integration bugs eat time, hardcode the checkpoint path and strip any remaining config flexibility (multi-model switching, config-driven vertical profiles per PMD Section 11.1) — none of that is needed for a single demo run.

---

### Block 2.3 (H32–H36): Temporal Smoothing & Confidence Display
**Tasks**
- Tune the GRU sliding-window behavior for visibly smoother output vs. raw single-frame flicker (this is the demo's core "wow" moment — showing the PMD's KR2.2 flicker-reduction concept qualitatively, not quantitatively)
- Add a readable on-screen confidence display (probability bar or percentage) next to the top predicted class
- Optional: add a simple "raw single-frame vs. temporally-smoothed" toggle for a compelling side-by-side demo moment

**Acceptance Criteria**
- Visually, the smoothed output changes less erratically frame-to-frame than a single-frame-only baseline run side-by-side or sequentially
- Confidence value is displayed and updates live

**Blocking Dependencies:** Block 2.2 working inference pipeline

**Fallback if overrun:** Drop the raw-vs-smoothed toggle (nice-to-have) and ship only the smoothed output with a confidence readout — this is the minimum viable version of this block.

---

### Block 2.4 (H36–H40): Live Webcam Integration — GO/NO-GO 5
**Tasks**
- Point the full pipeline at a live webcam feed instead of a pre-recorded file
- Confirm usable frame rate (does not need to hit the PMD's 30 FPS edge NFR — needs to be smooth enough for a live demo, roughly 5–15 FPS on CPU/GPU dev hardware is acceptable)
- Handle basic runtime hiccups (webcam permission, resolution mismatch)

**Acceptance Criteria**
- Live webcam feed produces classification + confidence overlay updating in near-real-time with no crashes over a 2-minute continuous run

**GO/NO-GO 5 Decision (H40):** If live webcam integration is unstable or too slow to be demo-safe by H40, **lock in pre-recorded video as the sole demo modality** and do not attempt further live-webcam debugging. A reliable pre-recorded demo beats a flaky live one — this decision is pre-approved and should be made without re-litigating in the moment.

**Blocking Dependencies:** Block 2.3 complete

**Fallback if overrun:** Pre-recorded-clip-only demo (see GO/NO-GO 5 decision above). This is a fully acceptable outcome, not a failure — flag it plainly in Section 7.

---

### Block 2.5 (H40–H44): Bug Bash & Edge-Case Hardening
**Tasks**
- Test and harden: no face in frame, extreme pose, low light, brief occlusion (hand over face) — the PMD's Section 8.1 edge cases, scoped down to "doesn't crash" rather than "handled gracefully with full state machine"
- Fix any crashes found; do not add new features
- Run the demo script (Section 6 below) end-to-end at least once as a dry run

**Acceptance Criteria**
- The pipeline does not crash on any of the four edge cases above; degraded output (e.g., "no face detected" text, or a frozen last-known state) is acceptable — a hard crash is not
- One full dry run of the demo script completes without intervention

**Blocking Dependencies:** Block 2.4 (or its fallback) locked in

**Fallback if overrun:** Prioritize only the "no face in frame" case if time is short — it is the single most likely failure mode to occur live on camera (presenter stepping out of frame, camera angle shift). Other edge cases can be verbally caveated during the demo if not hardened in time.

---

### Block 2.6 (H44–H46): Final Demo Rehearsal
**Tasks**
- Run the full demo script (Section 6) twice, back-to-back, exactly as it will be presented
- Time the demo; confirm it fits whatever presentation slot is expected
- Prepare a one-slide (or verbal) summary of what was cut and why (feeds directly from Section 1.2 and Section 7)

**Acceptance Criteria**
- Two consecutive successful rehearsal runs with no crashes and no manual code intervention between runs

**Blocking Dependencies:** Block 2.5 hardening complete

**Fallback if overrun:** If only one clean rehearsal is achieved by H46, proceed to freeze anyway — do not extend into the freeze window chasing a second perfect run. The freeze deadline is harder than the rehearsal-count target.

---

### Block 2.7 (H46–H48): Demo Freeze
**No new code is written, merged, or run against the demo path after H46.** Any last-minute change must be a revert to a known-good prior state, never a new feature or fix under time pressure. This window is reserved exclusively for: environment stability checks (laptop charged, webcam positioned, backup recorded clip staged), restating the pre-recorded fallback path if live webcam is in use, and idle standby.

---

## 5. Risk Register & Time-Boxing Rules

| # | Risk | Time-Box | Pre-Committed Fallback Decision |
|---|---|---|---|
| R1 | Dataset access/licensing friction (AffectNet-wild access delayed or blocked) | 1.5 hrs (H0–H1:30) | Switch immediately to FER2013 or another instantly-downloadable public dataset. No re-attempt of the blocked source mid-sprint. |
| R2 | Full pipeline (Blocks 1.2–1.5) does not produce a working forward pass | Hard gate at H14 (GO/NO-GO 2) | Strip to the simplest version that runs: drop SAFM, drop GRU (single-frame classification only), reintroduce components only if time recovers in Day 2. |
| R3 | Training loss does not converge / diverges / NaNs | 1 restart allowed, decision point at H21 (feeds GO/NO-GO 3) | One restart with reduced LR/batch size, then let it run overnight regardless of early trajectory. If Day 2 checkpoint is still poor (GO/NO-GO 4), demo with a partially-trained model and narrate that transparently — do not retrain from scratch. |
| R4 | Live webcam inference is too slow or unstable for a reliable live demo | Hard gate at H40 (GO/NO-GO 5) | Lock in pre-recorded-video-only demo. This is a pre-approved, acceptable outcome, not a failure to escalate. |
| R5 | Dependency/environment conflicts (`timm`/`torch`/`mediapipe` version mismatches, CUDA driver issues) | 1 hr max per incident; if unresolved, escalate to a minimal fallback stack immediately | Fall back to CPU-only inference (accept lower FPS) or swap the conflicting library for a simpler equivalent (e.g., OpenCV Haar cascade instead of MediaPipe) rather than losing hours to environment debugging. |

**General time-boxing rule applied throughout:** every block above carries its own fallback; no block is permitted to silently overrun into the next block's time without an explicit, pre-approved fallback decision being invoked. The GO/NO-GO checkpoints in Section 2 are the mandatory checkpoints where this is enforced.

---

## 6. Minimum Viable Demo Script

**Setting:** Dev machine, single monitor or projected screen, webcam attached (or pre-recorded clip staged as primary/fallback per GO/NO-GO 5 outcome).

**Runbook (approx. 5–7 minutes on camera):**

1. **Open on the PMD framing (30 sec):** State in one sentence what FERD is (edge-first, temporally-aware emotion recognition) and what this demo is — a 48-hour architectural proof-of-concept, not the full Phase 1 MVP.
2. **Show the input (30 sec):** Either start the live webcam feed with the presenter's own face in frame, or play the staged pre-recorded clip. Narrate: "This is unprocessed video — no cloud calls, running entirely on this machine."
3. **Show single-frame-only output first, if the toggle from Block 2.3 was built (60 sec):** Point out visible flicker/instability in the per-frame classification as the presenter holds a neutral expression — demonstrating the exact problem described in PMD Section 3.1 (#2, absence of temporal modeling).
4. **Switch to the full ST-ViT-GRU temporal pipeline (90 sec):** Same input, now with the GRU sliding window active. Narrate that the prediction is visibly steadier. Point at the on-screen confidence score.
5. **Trigger an occlusion moment on camera (60 sec):** Presenter partially covers their mouth with a hand. Narrate that the SAFM module down-weights the occluded region rather than failing outright — point to the mask visualization if built, or simply note the classification continues functioning (does not crash or output garbage).
6. **Trigger a "no face in frame" moment (30 sec):** Presenter steps briefly out of frame or turns fully away. Confirm the system displays an explicit "no face detected" state rather than crashing or guessing — this maps directly to PMD Section 8.1's edge-case handling philosophy, simplified for this sprint.
7. **Close with the honest scope statement (60 sec):** State plainly, using Section 1.2 and Section 7 of this document, what was simplified to hit 48 hours (heuristic SAFM instead of learned, small fine-tuning dataset, no quantization/edge export, no full benchmark validation) and what the next step would be to turn this into the PMD's actual Phase 1 MVP.

**What "success" looks like on camera:**
- The pipeline runs live, on this specific machine, with no crashes, for the full script.
- The temporally-smoothed output is visibly steadier than the single-frame baseline (even if not rigorously measured).
- The system fails gracefully (explicit "no face" state) rather than crashing when the presenter steps out of frame.
- The audience leaves understanding that this validates the *architectural shape* of ST-ViT-GRU, not its production-grade accuracy or latency claims.

---

## 7. Post-Sprint Handoff Notes

### 7.1 Technical debt / shortcuts taken to hit the 48-hour window

| Shortcut | What Was Actually Done | What Must Be Revisited Before Phase 1 |
|---|---|---|
| SAFM module | Heuristic (landmark visibility + local contrast), not a trained gating network | Replace with the PMD's actual trainable SAFM (`safm/attention_mask.py`) per Section 6.1/7 architecture spec |
| Encoder training | Pretrained MobileViT-XS, frozen or lightly fine-tuned; not trained on FERD's target domain (AffectNet-wild + DFEW) | Full teacher training per PMD Section 4.1 and CI/CD Stage 3 (Section 10.2) |
| Training data | Small public subset (FER2013-scale), not the licensed AffectNet-wild + DFEW combined benchmark | Formal data licensing and DVC-versioned ingestion per PMD Section 7 |
| Temporal fine-tuning data | If a static-image dataset was used, 16-frame "windows" were synthesized by repeating/lightly augmenting single frames — the GRU was **not** trained on genuine temporal dynamics | Source or license genuine video-sequence emotion data before claiming any real micro-expression or temporal-stability capability |
| Accuracy/robustness validation | No formal evaluation against AffectNet-wild/DFEW holdout, no occlusion-robustness benchmark, no CASME II/SAMM validation | Full KR1.1/KR2.1/KR2.3 evaluation per PMD Section 4.3 before any accuracy claim is made externally |
| Latency/FPS | Measured informally on dev-machine CPU/GPU only; no edge/NPU hardware involved | Full NFR validation on reference edge targets (Qualcomm QCS6490, Apple A16) per PMD Section 6.4/KR1.2 |
| Model compression | None — no PTQ, no QAT, no distillation, no export | Entire compression/export pipeline (PMD Section 7, `compression/`, `export/`) is unbuilt and required for any deployment claim |
| Multi-face / session management | Single-subject only, no session state machine | Build PMD Section 6.3/FERD-F07 session manager before any multi-subject use case |
| Compliance/privacy posture | Not assessed — this is a local dev-machine demo only | Full privacy review (KR3.1) required before any healthcare-adjacent claim or pilot conversation |

### 7.2 What this demo does and does not prove

**Proves:** the ST-ViT-GRU architectural shape (spatial encoder → attention masking → temporal head → classification) is implementable and wireable end-to-end, and that the concept of temporal smoothing measurably reduces frame-to-frame flicker versus a single-frame baseline, even on an undertrained model.

**Does not prove:** any of the PMD's quantitative OKRs (Section 4.3) — accuracy, latency, robustness-under-occlusion delta, or micro-expression recall. None of these should be cited as validated by this sprint's output. Any stakeholder communication about this demo should explicitly frame it as an architectural proof-of-concept, not a benchmark result.

### 7.3 Recommended immediate next steps into Phase 1

1. Scope and kick off formal AffectNet-wild + DFEW licensing/ingestion (PMD Section 5.2 cost line item)
2. Stand up the actual training cluster infrastructure (PMD Section 7, Docker + Kubernetes) rather than continuing on a single dev machine
3. Replace the heuristic SAFM with the trainable version and validate against an occlusion benchmark subset
4. Re-baseline effort/timeline estimates for Phase 1 (Months 1–4) using this sprint's actual wall-clock experience as a sanity check on the PMD's existing estimates
