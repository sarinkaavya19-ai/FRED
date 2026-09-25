"""
Block 2.3: Temporal Smoothing & Confidence Display
Enhanced demo showing GRU temporal smoothing vs raw single-frame predictions.

SPRINT-SIMPLIFICATION: Shows qualitative smoothing effect with fresh model.
Full PMD: Would use trained model with genuine temporal dynamics.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import create_streaming_pipeline, StreamingFERDPipeline
from preprocessing.face_align import FaceAligner, FaceAlignmentResult

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
EMOJIS = ["😠", "🤢", "😨", "😊", "😢", "😲", "😐"]


class TemporalSmoothingDemo:
    """Demo showing temporal smoothing with GRU vs single-frame baseline."""
    
    def __init__(self):
        self.model = None
        self.face_aligner = None
        self.running = False
        
        # For single-frame baseline (no temporal context)
        self.single_frame_buffer = deque(maxlen=1)
        
        # Smoothing history for visualization
        self.pred_history = deque(maxlen=30)
        self.conf_history = deque(maxlen=30)
        
    def initialize(self):
        """Initialize model and face aligner."""
        print("Initializing pipeline...")
        self.model = create_streaming_pipeline(
            encoder_variant='mobilevit_xs',
            encoder_pretrained=True,
            encoder_frozen=True,
            use_safm=True,
        )
        self.model.eval()
        
        print("Initializing face aligner...")
        self.face_aligner = FaceAligner(running_mode='VIDEO')
        
    def get_single_frame_prediction(self, frame):
        """Get prediction without temporal context (baseline)."""
        align_result = self.face_aligner.align(frame)
        if not align_result.success:
            return None
        
        # Prepare tensors
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
        
        # Single frame through encoder only (no GRU)
        with torch.no_grad():
            if self.model.safm is not None:
                masked_frame, _ = self.model.safm(tensor, landmarks, visibility)
            else:
                masked_frame = tensor
            
            embed = self.model.encoder(masked_frame)  # (1, 256)
            # Direct classification without GRU
            logits = self.model.classifier.classifier(embed)  # (1, 7)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            
            pred = probs.argmax().item()
            confidence = probs.max().item()
            
        return {
            'pred': pred,
            'conf': confidence,
            'probs': probs.cpu().numpy(),
            'bbox': align_result.bbox
        }
    
    def get_temporal_prediction(self, frame):
        """Get prediction with GRU temporal context."""
        align_result = self.face_aligner.align(frame)
        if not align_result.success:
            return None
        
        # Prepare tensors
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
        
        # Full pipeline with GRU
        with torch.no_grad():
            result = self.model.step(tensor, landmarks, visibility)
        
        pred_idx = result['pred_class'].item()
        confidence = result['confidence'].item()
        probs = result['probs'].squeeze(0).cpu().numpy()
        
        return {
            'pred': pred_idx,
            'conf': confidence,
            'probs': probs,
            'bbox': align_result.bbox
        }
    
    def draw_confidence_bars(self, frame, probs, x, y, width=200, height=18):
        """Draw horizontal probability bars for all 7 classes."""
        bar_h = height // 7
        for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
            bar_y = y + i * (bar_h + 2)
            
            # Background
            cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + bar_h), (40, 40, 40), -1)
            
            # Probability fill
            fill_w = int(width * prob)
            color = (0, 255, 0) if prob > 0.5 else (0, 200, 255) if prob > 0.3 else (0, 100, 255)
            cv2.rectangle(frame, (x, bar_y), (x + fill_w, bar_y + bar_h), color, -1)
            
            # Label
            label = f"{EMOJIS[i]} {emotion}: {prob:.0%}"
            cv2.putText(frame, label, (x + width + 10, bar_y + bar_h - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    def draw_main_prediction(self, frame, pred, conf, bbox, x, y, label_prefix=""):
        """Draw main emotion prediction with confidence."""
        if pred is None:
            return
        
        emotion = EMOTIONS[pred]
        emoji = EMOJIS[pred]
        
        # Main label
        main_text = f"{label_prefix}{emoji} {emotion}: {conf:.1%}"
        (tw, th), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Background
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
        color = (0, 255, 0) if conf > 0.5 else (0, 200, 255) if conf > 0.3 else (0, 100, 255)
        cv2.putText(frame, main_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    def draw_history_plot(self, frame, history, x, y, width=300, height=100, color=(0, 255, 0)):
        """Draw prediction confidence history as a plot."""
        if len(history) < 2:
            return
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)
        
        # Plot confidence history
        pts = []
        for i, val in enumerate(history):
            px = x + int(i * width / max(1, len(history) - 1))
            py = y + height - int(val * height)
            pts.append((px, py))
        
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, 2)
        
        # Current value
        cv2.putText(frame, f"Conf: {history[-1]:.1%}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def run_webcam(self):
        """Run demo with webcam."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nStarting temporal smoothing demo...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Screenshot")
        print("  't' - Toggle single-frame / temporal mode")
        print("  'h' - Toggle history plot")
        print("  'b' - Toggle bbox")
        
        show_temporal = True  # True = temporal (GRU), False = single-frame
        show_history = True
        show_bbox = True
        
        self.running = True
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Get predictions
            if show_temporal:
                result = self.get_temporal_prediction(frame)
                mode_text = "TEMPORAL (GRU)"
                mode_color = (0, 255, 0)
            else:
                result = self.get_single_frame_prediction(frame)
                mode_text = "SINGLE-FRAME"
                mode_color = (0, 165, 255)
            
            # Update history
            if result:
                self.pred_history.append(result['pred'])
                self.conf_history.append(result['conf'])
            
            # Draw bbox
            if show_bbox and result and result['bbox']:
                x, y, bw, bh = result['bbox']
                cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
            # Draw main prediction
            if result:
                self.draw_main_prediction(display_frame, result['pred'], result['conf'], 
                                        result['bbox'], 10, 40, label_prefix="")
            
            # Draw confidence bars
            if result:
                self.draw_confidence_bars(display_frame, result['probs'], 10, 80)
            
            # Draw history plot
            if show_history and self.conf_history:
                self.draw_history_plot(display_frame, self.conf_history, 
                                     w - 320, 10, width=300, height=120)
            
            # Draw mode indicator
            cv2.putText(display_frame, mode_text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)
            
            # Draw legend
            cv2.putText(display_frame, "Controls: q=quit s=shot t=toggle h=history b=bbox",
                       (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow("FERD Temporal Smoothing Demo", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"temporal_demo_{int(cv2.getTickCount())}.jpg", display_frame)
                print("Screenshot saved!")
            elif key == ord('t'):
                show_temporal = not show_temporal
                print(f"Mode: {'TEMPORAL' if show_temporal else 'SINGLE-FRAME'}")
            elif key == ord('h'):
                show_history = not show_history
            elif key == ord('b'):
                show_bbox = not show_bbox
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video(self, video_path):
        """Run demo on video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return
        
        print(f"Processing video: {video_path}")
        
        show_temporal = True
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                # Show last frame when paused
                pass
            
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            if show_temporal:
                result = self.get_temporal_prediction(frame)
            else:
                result = self.get_single_frame_prediction(frame)
            
            if result:
                self.conf_history.append(result['conf'])
            
            # Draw results
            if show_bbox and result and result['bbox']:
                x, y, bw, bh = result['bbox']
                cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
            if result:
                self.draw_main_prediction(display_frame, result['pred'], result['conf'],
                                        result['bbox'], 10, 40)
                self.draw_confidence_bars(display_frame, result['probs'], 10, 80)
            
            if show_history and self.conf_history:
                self.draw_history_plot(display_frame, self.conf_history, w - 320, 10)
            
            mode_text = "TEMPORAL (GRU)" if show_temporal else "SINGLE-FRAME"
            mode_color = (0, 255, 0) if show_temporal else (0, 165, 255)
            cv2.putText(display_frame, mode_text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            
            cv2.imshow("FERD Temporal Smoothing Demo", display_frame)
            
            key = cv2.waitKey(0 if paused else 30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p') or key == ord(' '):
                paused = not paused
            elif key == ord('t'):
                show_temporal = not show_temporal
                print(f"Mode: {'TEMPORAL' if show_temporal else 'SINGLE-FRAME'}")
            elif key == ord('h'):
                show_history = not show_history
            elif key == ord('b'):
                show_bbox = not show_bbox
        
        cap.release()
        cv2.destroyAllWindows()
    
    def cleanup(self):
        if self.face_aligner:
            self.face_aligner.close()
        cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("BLOCK 2.3: TEMPORAL SMOOTHING & CONFIDENCE DISPLAY")
    print("=" * 60)
    
    demo = TemporalSmoothingDemo()
    demo.initialize()
    
    # Check for test video
    test_video = "test_emotion_clip.mp4"
    if Path(test_video).exists():
        print(f"Found test video: {test_video}")
        choice = input("Run on (w)ebcam or (v)ideo? [w/v]: ").strip().lower()
        if choice == 'v':
            demo.run_video(test_video)
        else:
            demo.run_webcam()
    else:
        print("No test video found, running webcam...")
        demo.run_webcam()
    
    demo.cleanup()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()