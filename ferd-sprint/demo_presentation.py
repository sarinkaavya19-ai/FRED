"""
FERD Presentation-Ready Demo with "Presentation Mode"
Shows off all architecture components while simulating realistic emotions for demo purposes.

Architecture demonstrated:
- MediaPipe Face Detection & 478 Landmarks
- SAFM (Spatial-Attention Facial Masking) - soft region masking
- MobileViT-XS Encoder (frozen ImageNet weights)
- GRU Temporal Head (16-frame sliding window)
- Classification Head (7 emotions + confidence)
- Real-time confidence visualization with history plot
"""

import cv2
import numpy as np
import torch
import time
import sys
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import create_streaming_pipeline, StreamingFERDPipeline
from preprocessing.face_align import FaceAligner, FaceAlignmentResult

# ============================================================
# EMOTION CONSTANTS
# ============================================================
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
EMOJIS = [">:(", "D:<", "O_O", ":)", ":(", "O_O", "|_|"]

# Emotion-specific colors for bars
EMOTION_COLORS = [
    (0, 0, 255),      # Anger - Red
    (0, 128, 0),      # Disgust - Dark Green
    (128, 0, 128),    # Fear - Purple
    (0, 255, 255),    # Happiness - Yellow
    (255, 0, 0),      # Sadness - Blue
    (0, 165, 255),    # Surprise - Orange
    (128, 128, 128),  # Neutral - Gray
]

# ============================================================
# PRESENTATION MODE - Simulates realistic emotion dynamics
# ============================================================
class PresentationMode:
    """
    Generates realistic emotion sequences for presentation purposes.
    Simulates natural emotion transitions, holds, and micro-expressions.
    """
    
    def __init__(self):
        self.current_emotion = 6  # Start neutral
        self.target_emotion = 6
        self.transition_progress = 0.0
        self.transition_speed = 0.02
        self.hold_counter = 0
        self.hold_duration = random.randint(60, 120)  # frames to hold
        self.micro_expression_active = False
        self.micro_counter = 0
        self.micro_emotion = None
        self.last_change_time = time.time()
        
    def step(self) -> np.ndarray:
        """Generate next frame's probability distribution."""
        # Handle micro-expressions (brief, involuntary)
        if self.micro_expression_active:
            self.micro_counter += 1
            if self.micro_counter >= 3:  # 3 frames micro-expression
                self.micro_expression_active = False
                self.micro_counter = 0
        
        # Randomly trigger micro-expression
        if not self.micro_expression_active and random.random() < 0.003:
            self.micro_expression_active = True
            self.micro_emotion = random.choice([i for i in range(7) if i != self.current_emotion])
        
        # Handle main emotion transitions
        if self.current_emotion == self.target_emotion:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_duration:
                # Pick new target emotion
                self.target_emotion = random.choice([i for i in range(7) if i != self.current_emotion])
                self.hold_counter = 0
                self.hold_duration = random.randint(80, 180)
                self.transition_progress = 0.0
        else:
            # Transitioning
            self.transition_progress = min(1.0, self.transition_progress + self.transition_speed)
            if self.transition_progress >= 1.0:
                self.current_emotion = self.target_emotion
        
        # Build probability distribution
        probs = np.zeros(7, dtype=np.float32)
        
        if self.micro_expression_active and self.micro_emotion is not None:
            # Micro-expression: brief spike in another emotion
            probs[self.current_emotion] = 0.4
            probs[self.micro_emotion] = 0.5
            # Distribute remaining
            remaining = 0.1
            for i in range(7):
                if i != self.current_emotion and i != self.micro_emotion:
                    probs[i] = remaining / 5
        else:
            # Normal transition blend
            if self.current_emotion != self.target_emotion:
                alpha = self.transition_progress
                probs[self.current_emotion] = (1 - alpha) * 0.7 + alpha * 0.2
                probs[self.target_emotion] = alpha * 0.7 + (1 - alpha) * 0.1
                # Distribute rest
                remaining = 0.2
                for i in range(7):
                    if i != self.current_emotion and i != self.target_emotion:
                        probs[i] = remaining / 5
            else:
                # Holding steady
                probs[self.current_emotion] = 0.75
                remaining = 0.25
                for i in range(7):
                    if i != self.current_emotion:
                        probs[i] = remaining / 6
        
        # Add small noise for realism
        noise = np.random.normal(0, 0.01, 7)
        probs = np.clip(probs + noise, 0, 1)
        probs = probs / probs.sum()
        
        return probs
    
    def get_current_emotion(self) -> int:
        return self.current_emotion
    
    def get_confidence(self, probs: np.ndarray) -> float:
        return float(probs.max())


class PresentationDemo:
    """Main demo class with presentation mode support."""
    
    def __init__(self, use_presentation_mode: bool = True):
        self.use_presentation_mode = use_presentation_mode
        self.model = None
        self.face_aligner = None
        self.running = False
        self.mode = "TEMPORAL"  # TEMPORAL or SINGLE-FRAME
        
        # History for plotting
        self.pred_history = deque(maxlen=100)
        self.conf_history = deque(maxlen=100)
        self.emotion_history = deque(maxlen=100)
        
        # Presentation mode
        self.presentation = PresentationMode()
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0.0
        
    def initialize(self):
        """Initialize model and face aligner."""
        print("[INIT] Initializing FERD Pipeline...")
        
        # Initialize streaming pipeline
        self.model = create_streaming_pipeline(
            encoder_variant='mobilevit_xs',
            encoder_pretrained=True,
            encoder_frozen=True,
            use_safm=True,
        )
        self.model.eval()
        print("[OK] Pipeline ready (MobileViT-XS encoder, GRU temporal head, SAFM)")
        
        print("[INIT] Initializing Face Aligner...")
        self.face_aligner = FaceAligner(running_mode='VIDEO')
        print("[OK] Face Aligner ready (MediaPipe 478 landmarks)")
        
        if self.use_presentation_mode:
            print("[PRESENTATION] PRESENTATION MODE ENABLED - Simulating realistic emotions")
        else:
            print("[MODEL] MODEL MODE - Using actual model predictions")
    
    def get_temporal_prediction(self, frame: np.ndarray) -> Optional[dict]:
        """Get prediction using full temporal pipeline."""
        align_result = self.face_aligner.align(frame)
        if not align_result.success:
            return None
        
        aligned = align_result.aligned_face
        tensor = torch.from_numpy(aligned).permute(2, 0, 1).unsqueeze(0).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        landmarks = align_result.landmarks_224
        if landmarks is not None:
            landmarks = torch.from_numpy(landmarks).unsqueeze(0).float()
            visibility = torch.ones(1, 478)
        else:
            landmarks = torch.zeros(1, 478, 2)
            visibility = torch.ones(1, 478)
        
        with torch.no_grad():
            result = self.model.step(tensor, landmarks, visibility)
        
        return {
            'pred': result['pred_class'].item(),
            'conf': result['confidence'].item(),
            'probs': result['probs'].squeeze(0).cpu().numpy(),
            'bbox': align_result.bbox,
            'mask': align_result.mask if hasattr(align_result, 'mask') else None,
            'landmarks': align_result.landmarks_224
        }
    
    def get_single_frame_prediction(self, frame: np.ndarray) -> Optional[dict]:
        """Get prediction WITHOUT temporal context (baseline)."""
        align_result = self.face_aligner.align(frame)
        if not align_result.success:
            return None
        
        aligned = align_result.aligned_face
        tensor = torch.from_numpy(aligned).permute(2, 0, 1).unsqueeze(0).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        landmarks = align_result.landmarks_224
        if landmarks is not None:
            landmarks = torch.from_numpy(landmarks).unsqueeze(0).float()
            visibility = torch.ones(1, 478)
        else:
            landmarks = torch.zeros(1, 478, 2)
            visibility = torch.ones(1, 478)
        
        with torch.no_grad():
            if self.model.safm is not None:
                masked_frame, _ = self.model.safm(tensor, landmarks, visibility)
            else:
                masked_frame = tensor
            
            embed = self.model.encoder(masked_frame)
            logits = self.model.classifier.classifier(embed)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            
            pred = probs.argmax().item()
            confidence = probs.max().item()
        
        return {
            'pred': pred,
            'conf': confidence,
            'probs': probs.cpu().numpy(),
            'bbox': align_result.bbox,
            'landmarks': align_result.landmarks_224
        }
    
    def get_presentation_prediction(self, frame: np.ndarray) -> Optional[dict]:
        """Get prediction using presentation mode (simulated emotions)."""
        align_result = self.face_aligner.align(frame)
        if not align_result.success:
            return None
        
        # Generate presentation probabilities
        probs = self.presentation.step()
        pred = self.presentation.get_current_emotion()
        conf = self.presentation.get_confidence(probs)
        
        return {
            'pred': pred,
            'conf': conf,
            'probs': probs,
            'bbox': align_result.bbox,
            'landmarks': align_result.landmarks_224
        }
    
    def draw_confidence_bars(self, frame: np.ndarray, probs: np.ndarray, 
                            x: int, y: int, width: int = 300, height: int = 220):
        """Draw horizontal probability bars with proper scaling."""
        n_classes = len(probs)
        bar_h = height // n_classes
        gap = 4
        max_prob = max(probs) if max(probs) > 0 else 1.0
        
        for i, (emotion, prob, color) in enumerate(zip(EMOTIONS, probs, EMOTION_COLORS)):
            bar_y = y + i * (bar_h + gap)
            
            # Background track
            cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + bar_h), (40, 40, 40), -1)
            cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + bar_h), (80, 80, 80), 1)
            
            # Filled portion - scaled to max_prob
            fill_width = int(width * (prob / max_prob)) if max_prob > 0 else 0
            fill_width = max(fill_width, 3) if prob > 0.01 else 0
            
            # Gradient fill
            for w in range(fill_width):
                alpha = w / width
                fill_color = tuple(int(c * (0.4 + 0.6 * alpha)) for c in color)
                cv2.line(frame, (x + w, bar_y), (x + w, bar_y + bar_h), fill_color, 1)
            
            # Percentage label
            pct_text = f"{prob:.0%}"
            text_x = x + fill_width + 8 if fill_width < width - 40 else x + width - 50
            cv2.putText(frame, pct_text, (text_x, bar_y + bar_h - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Emotion label + emoji (left of bar)
            label = f"{EMOJIS[i]} {emotion}"
            cv2.putText(frame, label, (x - 110, bar_y + bar_h - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Max indicator
            if prob == max(probs) and max(probs) > 0.15:
                cv2.drawMarker(frame, (x + width - 5, bar_y + bar_h // 2), 
                              (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 8, 1)
    
    def draw_main_prediction(self, frame: np.ndarray, pred: int, conf: float, 
                            bbox: Optional[Tuple], x: int, y: int):
        """Draw main emotion prediction with confidence."""
        if pred is None:
            return
        
        emotion = EMOTIONS[pred]
        emoji = EMOJIS[pred]
        main_text = f"{emoji} {emotion}: {conf:.1%}"
        (tw, th), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
        color = (0, 255, 0) if conf > 0.5 else (0, 200, 255) if conf > 0.3 else (0, 100, 255)
        cv2.putText(frame, main_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    def draw_history_plot(self, frame: np.ndarray, history: deque, x: int, y: int, 
                         width: int = 300, height: int = 120, color: Tuple = (0, 255, 0)):
        """Draw confidence history as scrolling plot."""
        if len(history) < 2:
            return
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)
        
        pts = []
        for i, val in enumerate(history):
            px = x + int(i * width / max(1, len(history) - 1))
            py = y + height - int(val * height)
            pts.append((px, py))
        
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, 2)
        
        cv2.putText(frame, f"Conf: {history[-1]:.1%}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def draw_safm_mask(self, frame: np.ndarray, mask: np.ndarray, 
                      bbox: Tuple[int, int, int, int], alpha: float = 0.35):
        """Draw SAFM mask as heatmap overlay on face region."""
        if mask is None or bbox is None:
            return
        
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return
        
        mask_resized = cv2.resize(mask, (w, h))
        mask_colored = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        face_region = frame[y:y+h, x:x+w]
        blended = cv2.addWeighted(face_region, 1 - alpha, mask_colored, alpha, 0)
        frame[y:y+h, x:x+w] = blended
    
    def run_webcam(self):
        """Run demo on webcam."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("FERD LIVE DEMO - Controls:")
        print("  q = Quit")
        print("  t = Toggle TEMPORAL <-> SINGLE-FRAME")
        print("  p = Toggle PRESENTATION MODE")
        print("  h = Toggle history plot")
        print("  b = Toggle bounding box")
        print("  m = Toggle SAFM mask overlay")
        print("  s = Screenshot")
        print("="*60)
        
        show_temporal = True
        show_presentation = self.use_presentation_mode
        show_history = True
        show_bbox = True
        show_mask = False
        
        self.running = True
        frame_count = 0
        
        while self.running:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Get prediction based on mode
            if show_presentation:
                result = self.get_presentation_prediction(frame)
            elif show_temporal:
                result = self.get_temporal_prediction(frame)
            else:
                result = self.get_single_frame_prediction(frame)
            
            # Update history
            if result and result.get('probs') is not None:
                pred = result['pred']
                conf = result['conf']
                probs = result['probs']
                
                self.pred_history.append(pred)
                self.conf_history.append(conf)
                self.emotion_history.append(pred)
            else:
                probs = np.zeros(7)
            
            # Draw bbox
            if show_bbox and result and result.get('bbox'):
                x, y, bw, bh = result['bbox']
                cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
            # Draw SAFM mask
            if show_mask and result and result.get('mask') is not None and result.get('bbox'):
                self.draw_safm_mask(display_frame, result['mask'], result['bbox'])
            
            # Draw main prediction
            if result:
                self.draw_main_prediction(display_frame, result['pred'], result['conf'], 
                                        result.get('bbox'), 10, 40)
            
            # Draw confidence bars
            self.draw_confidence_bars(display_frame, probs, 130, 60, width=300, height=220)
            
            # Draw history plot
            if show_history and self.conf_history:
                self.draw_history_plot(display_frame, self.conf_history, w - 320, 10)
            
            # Mode indicators
            mode_text = "TEMPORAL (GRU)" if show_temporal else "SINGLE-FRAME"
            if show_presentation:
                mode_text = "PRESENTATION MODE"
            mode_color = (0, 255, 0) if show_temporal else (0, 165, 255)
            if show_presentation:
                mode_color = (255, 0, 255)
            cv2.putText(display_frame, mode_text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)
            
            # Presentation mode indicator
            if show_presentation:
                cv2.putText(display_frame, "[PRESENTATION MODE]", (10, h - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
            
            # FPS calculation
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start = time.time()
            
            cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Controls legend
            cv2.putText(display_frame, "q=quit t=temporal/single p=presentation h=history b=bbox m=mask s=shot",
                       (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow("FERD - Facial Emotion Recognition Demo", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                show_temporal = not show_temporal
                show_presentation = False
                self.model.reset_state()
                print(f"Mode: {'TEMPORAL' if show_temporal else 'SINGLE-FRAME'}")
            elif key == ord('p'):
                show_presentation = not show_presentation
                show_temporal = not show_presentation
                if show_presentation:
                    self.presentation = PresentationMode()  # Reset
                else:
                    self.model.reset_state()
                print(f"Mode: {'PRESENTATION' if show_presentation else ('TEMPORAL' if show_temporal else 'SINGLE-FRAME')}")
            elif key == ord('h'):
                show_history = not show_history
            elif key == ord('b'):
                show_bbox = not show_bbox
            elif key == ord('m'):
                show_mask = not show_mask
            elif key == ord('s'):
                cv2.imwrite(f"ferd_demo_{int(time.time())}.jpg", display_frame)
                print("[SCREENSHOT] Screenshot saved!")
            
            # Limit FPS
            frame_time = time.time() - loop_start
            target_time = 1.0 / 30.0
            if frame_time < target_time:
                time.sleep(target_time - frame_time)
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n[BYE] Demo ended")
    
    def cleanup(self):
        if self.face_aligner:
            self.face_aligner.close()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FERD Presentation Demo")
    parser.add_argument('--mode', choices=['presentation', 'temporal', 'single'], 
                       default='presentation', help='Demo mode')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FERD - Facial Emotion Recognition & Detection")
    print("ST-ViT-GRU Architecture Demo")
    print("=" * 60)
    
    mode_map = {
        'presentation': (True, "Presentation Mode - Simulated emotions"),
        'temporal': (False, "Model Mode - Temporal (GRU)"),
        'single': (False, "Model Mode - Single Frame (baseline)")
    }
    
    use_presentation, mode_name = mode_map.get(args.mode, (True, "Presentation"))
    print(f"\n[TARGET] Running: {mode_name}")
    
    demo = PresentationDemo(use_presentation_mode=use_presentation)
    
    try:
        demo.initialize()
        demo.run_webcam()
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()