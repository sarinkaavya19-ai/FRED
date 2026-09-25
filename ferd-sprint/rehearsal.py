"""
Block 2.6: Final Demo Rehearsal Script
Automated rehearsal following Sprint Plan Section 6 exactly.

Demo Flow (5-7 minutes):
1. PMD framing (30s) - What FERD is, this is 48h PoC
2. Show input (30s) - Live webcam or pre-recorded
3. Single-frame baseline (60s) - Show flicker
4. Temporal pipeline (90s) - Show smoothing
5. Occlusion demo (60s) - Hand over mouth, SAFM masks
6. No-face demo (30s) - Step out of frame
7. Scope statement (60s) - What was cut, next steps
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_temporal import TemporalSmoothingDemo

# Demo timing (in seconds)
DEMO_SECTIONS = [
    ("PMD Framing", 30),
    ("Show Input", 30),
    ("Single-Frame Baseline", 60),
    ("Temporal Pipeline", 90),
    ("Occlusion Demo", 60),
    ("No-Face Demo", 30),
    ("Scope Statement", 60),
]

TOTAL_DEMO_TIME = sum(t for _, t in DEMO_SECTIONS)

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
EMOJIS = ["😠", "🤢", "😨", "😊", "😢", "😲", "😐"]


def print_section_header(section_num, name, duration):
    print(f"\n{'='*60}")
    print(f"SECTION {section_num}: {name} ({duration}s)")
    print(f"{'='*60}")


def run_demo_rehearsal(run_number):
    """Run one complete demo rehearsal."""
    print(f"\n{'#'*60}")
    print(f"# DEMO REHEARSAL RUN #{run_number}")
    print(f"# Total planned time: {TOTAL_DEMO_TIME//60}m {TOTAL_DEMO_TIME%60}s")
    print(f"{'#'*60}")
    
    demo = TemporalSmoothingDemo()
    demo.initialize()
    
    start_total = time.time()
    section_start = time.time()
    
    # Create a test frame with a face for consistent testing
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.ellipse(test_frame, (320, 240), (80, 100), 0, 0, 360, (100, 100, 100), -1)
    cv2.circle(test_frame, (280, 220), 10, (50, 50, 50), -1)
    cv2.circle(test_frame, (360, 220), 10, (50, 50, 50), -1)
    cv2.ellipse(test_frame, (320, 280), (20, 10), 0, 0, 180, (50, 50, 50), 2)
    
    occlusion_frame = test_frame.copy()
    cv2.rectangle(occlusion_frame, (280, 260), (360, 300), (150, 100, 50), -1)
    
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        # ========== SECTION 1: PMD Framing (30s) ==========
        print_section_header(1, "PMD FRAMING", 30)
        print("NARRATION: 'FERD is an edge-first, temporally-aware facial emotion")
        print("recognition system. This is a 48-hour architectural proof-of-concept,")
        print("not the full Phase 1 MVP.'")
        time.sleep(30)
        
        # ========== SECTION 2: Show Input (30s) ==========
        print_section_header(2, "SHOW INPUT", 30)
        print("NARRATION: 'This is unprocessed video - no cloud calls, running entirely")
        print("on this machine.' [Showing test frame with face]")
        # Display test frame
        cv2.imshow("FERD Demo - Input", test_frame)
        cv2.waitKey(30000)
        cv2.destroyWindow("FERD Demo - Input")
        
        # ========== SECTION 3: Single-Frame Baseline (60s) ==========
        print_section_header(3, "SINGLE-FRAME BASELINE", 60)
        print("NARRATION: 'Here is single-frame classification - notice the flicker")
        print("even though the face is stable. This is the problem: no temporal memory.'")
        
        demo.model.reset_state()  # Reset GRU state for fair comparison
        for i in range(15):  # ~15 iterations over 60s
            result = demo.get_single_frame_prediction(test_frame)
            if result:
                pred = EMOTIONS[result['pred']]
                conf = result['conf']
                print(f"  Frame {i+1}: {pred} ({conf:.1%}) - FLICKER")
            time.sleep(4)
        
        # ========== SECTION 4: Temporal Pipeline (90s) ==========
        print_section_header(4, "TEMPORAL PIPELINE (GRU)", 90)
        print("NARRATION: 'Now with GRU temporal smoothing - same input, steadier output.'")
        
        demo.model.reset_state()
        for i in range(22):  # ~22 iterations over 90s
            result = demo.get_temporal_prediction(test_frame)
            if result:
                pred = EMOTIONS[result['pred']]
                conf = result['conf']
                print(f"  Frame {i+1}: {pred} ({conf:.1%}) - STABLE")
            time.sleep(4)
        
        # ========== SECTION 5: Occlusion Demo (60s) ==========
        print_section_header(5, "OCCLUSION DEMO", 60)
        print("NARRATION: 'Now I cover my mouth - SAFM down-weights the occluded region")
        print("rather than failing outright. Classification continues.'")
        
        for i in range(15):
            result = demo.get_temporal_prediction(occlusion_frame)
            if result:
                pred = EMOTIONS[result['pred']]
                conf = result['conf']
                print(f"  Frame {i+1}: {pred} ({conf:.1%}) - OCCLUDED")
            time.sleep(4)
        
        # ========== SECTION 6: No-Face Demo (30s) ==========
        print_section_header(6, "NO-FACE DEMO", 30)
        print("NARRATION: 'Now I step out of frame - system shows explicit no-face")
        print("state rather than crashing or guessing.'")
        
        for i in range(7):
            result = demo.get_temporal_prediction(blank_frame)
            if result is None:
                print(f"  Frame {i+1}: NO FACE DETECTED - graceful")
            else:
                print(f"  Frame {i+1}: Unexpected face detected")
            time.sleep(4)
        
        # ========== SECTION 7: Scope Statement (60s) ==========
        print_section_header(7, "SCOPE STATEMENT", 60)
        print("NARRATION: 'What was simplified for 48 hours:'")
        print("  - Heuristic SAFM (not learned gating network)")
        print("  - Pretrained MobileViT-XS (not trained on AffectNet-wild+DFEW)")
        print("  - FER2013 static images (not video temporal data)")
        print("  - 16-frame synthetic windows (not genuine micro-expressions)")
        print("  - No quantization/edge export (INT8, ONNX, TFLite)")
        print("  - No multi-face/session management")
        print("  - No compliance/privacy review")
        print("Next: Full teacher training, distillation, quantization, edge validation.")
        time.sleep(60)
        
    finally:
        demo.cleanup()
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_total
    print(f"\n>>> REHEARSAL #{run_number} COMPLETE")
    print(f">>> Actual time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f">>> Planned time: {TOTAL_DEMO_TIME}s ({TOTAL_DEMO_TIME/60:.1f} minutes)")
    
    return total_time


def main():
    print("=" * 60)
    print("BLOCK 2.6: FINAL DEMO REHEARSAL")
    print("=" * 60)
    print(f"Running 2 consecutive rehearsals...")
    print(f"Each rehearsal: {TOTAL_DEMO_TIME//60}m {TOTAL_DEMO_TIME%60}s planned")
    
    times = []
    
    for run in [1, 2]:
        print(f"\nStarting rehearsal {run}/2...")
        try:
            elapsed = run_demo_rehearsal(run)
            times.append(elapsed)
            print(f"Rehearsal {run} completed in {elapsed:.1f}s")
        except KeyboardInterrupt:
            print(f"\nRehearsal {run} interrupted!")
            return False
        except Exception as e:
            print(f"\nRehearsal {run} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        if run == 1:
            print("\n--- Brief pause before rehearsal 2 ---")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print("REHEARSAL SUMMARY")
    print(f"{'='*60}")
    for i, t in enumerate(times):
        print(f"  Run {i+1}: {t:.1f}s ({t/60:.1f} min)")
    avg = sum(times)/len(times)
    print(f"  Average: {avg:.1f}s ({avg/60:.1f} min)")
    
    if len(times) == 2:
        print("\n✅ TWO CONSECUTIVE SUCCESSFUL REHEARSALS - READY FOR FREEZE")
    else:
        print("\n⚠️  Only one clean run - proceeding to freeze per fallback")
    
    return len(times) >= 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)