"""
Demo Harness for FERD Sprint
Video/webcam capture loop with on-screen overlay rendering.

SPRINT-SIMPLIFICATION: Uses dummy/random outputs for UI development.
Full PMD: Integrates with trained StreamingFERDPipeline for real predictions.
"""

import cv2
import numpy as np
import torch
import time
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import create_streaming_pipeline, StreamingFERDPipeline
from preprocessing.face_align import FaceAligner, FaceAlignmentResult


@dataclass
class DemoConfig:
    """Configuration for demo harness."""
    # Input source
    source: str = "webcam"  # "webcam", "video", "image"
    video_path: str = ""
    image_path: str = ""
    webcam_index: int = 0
    
    # Display
    window_name: str = "FERD Demo - Emotion Recognition"
    show_fps: bool = True
    show_confidence: bool = True
    show_bbox: bool = True
    show_landmarks: bool = False
    show_mask: bool = False
    
    # Processing
    process_every_n_frames: int = 1  # Process every frame
    max_fps: float = 30.0
    
    # Model
    use_trained_model: bool = False
    checkpoint_path: str = ""
    
    # Colors (BGR)
    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_bg_color: Tuple[int, int, int] = (0, 0, 0)
    low_conf_color: Tuple[int, int, int] = (0, 165, 255)  # Orange
    no_face_color: Tuple[int, int, int] = (0, 0, 255)     # Red


class EmotionVisualizer:
    """Handles on-screen rendering of emotion predictions."""
    
    # PMD 7-class emotions
    EMOTIONS = [
        "Anger", "Disgust", "Fear", "Happiness", 
        "Sadness", "Surprise", "Neutral"
    ]
    
    # Emoji mapping for fun
    EMOJIS = {
        "Anger": "😠", "Disgust": "🤢", "Fear": "😨",
        "Happiness": "😊", "Sadness": "😢", 
        "Surprise": "😲", "Neutral": "😐"
    }
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.line_height = 25
    
    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  color: Tuple[int, int, int] = None, thickness: int = 2):
        """Draw bounding box around face."""
        if not self.config.show_bbox or bbox is None:
            return
        
        x, y, w, h = bbox
        color = color or self.config.bbox_color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    def draw_emotion_text(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          pred_class: int, confidence: float, 
                          all_probs: Optional[np.ndarray] = None):
        """Draw emotion label and confidence above bbox."""
        if not self.config.show_confidence or bbox is None:
            return
        
        x, y, w, h = bbox
        
        # Get emotion label
        emotion = self.EMOTIONS[pred_class] if 0 <= pred_class < 7 else "Unknown"
        emoji = self.EMOJIS.get(emotion, "")
        
        # Format text
        if self.config.show_confidence:
            label = f"{emoji} {emotion}: {confidence:.1%}"
        else:
            label = f"{emoji} {emotion}"
        
        # Determine color based on confidence
        if confidence < 0.4:
            text_color = self.config.low_conf_color
        else:
            text_color = self.config.text_color
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.font_thickness
        )
        
        # Position above bbox
        text_x = x
        text_y = y - 10
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_h - 5),
            (text_x + text_w + 5, text_y + baseline + 5),
            self.config.text_bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, label, (text_x, text_y),
            self.font, self.font_scale, text_color, self.font_thickness, cv2.LINE_AA
        )
        
        # Draw probability bars if provided
        if all_probs is not None and len(all_probs) == 7:
            self._draw_prob_bars(frame, x + w + 10, y, all_probs)
    
    def _draw_prob_bars(self, frame: np.ndarray, start_x: int, start_y: int, 
                        probs: np.ndarray, bar_width: int = 150, bar_height: int = 15):
        """Draw horizontal probability bars for all classes."""
        for i, (emotion, prob) in enumerate(zip(self.EMOTIONS, probs)):
            y = start_y + i * (bar_height + 5)
            
            # Background
            cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height),
                         (50, 50, 50), -1)
            
            # Probability bar
            fill_width = int(bar_width * prob)
            color = (0, 255, 0) if prob > 0.5 else (0, 255, 255) if prob > 0.3 else (0, 100, 255)
            cv2.rectangle(frame, (start_x, y), (start_x + fill_width, y + bar_height),
                         color, -1)
            
            # Label
            label = f"{emotion}: {prob:.1%}"
            cv2.putText(frame, label, (start_x + bar_width + 5, y + bar_height - 3),
                       self.font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    def draw_fps(self, frame: np.ndarray, fps: float):
        """Draw FPS counter."""
        if not self.config.show_fps:
            return
        
        label = f"FPS: {fps:.1f}"
        cv2.putText(frame, label, (10, 30), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    def draw_status(self, frame: np.ndarray, status: str, color: Tuple[int, int, int] = None):
        """Draw status message."""
        color = color or self.config.text_color
        cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                   self.font, 0.7, color, 2, cv2.LINE_AA)
    
    def draw_no_face(self, frame: np.ndarray):
        """Draw 'No face detected' message."""
        h, w = frame.shape[:2]
        label = "No Face Detected"
        (text_w, text_h), _ = cv2.getTextSize(label, self.font, 1.0, 2)
        x = (w - text_w) // 2
        y = (h + text_h) // 2
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y - text_h - 20), (w, y + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, label, (x, y), self.font, 1.0, self.config.no_face_color, 2, cv2.LINE_AA)
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, 
                       color: Tuple[int, int, int] = (255, 0, 0), radius: int = 1):
        """Draw facial landmarks."""
        if not self.config.show_landmarks or landmarks is None:
            return
        
        for pt in landmarks:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), radius, color, -1)
    
    def draw_mask_heatmap(self, frame: np.ndarray, mask: np.ndarray, 
                          bbox: Tuple[int, int, int, int], alpha: float = 0.4):
        """Draw SAFM mask as heatmap overlay on face region."""
        if not self.config.show_mask or mask is None or bbox is None:
            return
        
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return
        
        # Resize mask to bbox size
        mask_resized = cv2.resize(mask, (w, h))
        mask_colored = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay on face region
        face_region = frame[y:y+h, x:x+w]
        blended = cv2.addWeighted(face_region, 1 - alpha, mask_colored, alpha, 0)
        frame[y:y+h, x:x+w] = blended


class DemoHarness:
    """
    Main demo harness for FERD emotion recognition.
    
    Supports:
    - Webcam live feed
    - Video file playback
    - Single image
    - Dummy mode (no model, random predictions)
    - Trained model mode (with checkpoint)
    """
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.visualizer = EmotionVisualizer(config)
        
        # State
        self.running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Model (optional)
        self.pipeline: Optional[StreamingFERDPipeline] = None
        self.face_aligner: Optional[FaceAligner] = None
        self.use_model = config.use_trained_model and config.checkpoint_path
        
        # Dummy prediction state (for UI development without model)
        self.dummy_mode = not self.use_model
        self.dummy_pred_idx = 0
        self.dummy_confidence = 0.5
    
    def initialize_model(self):
        """Load trained model if available."""
        if not self.use_model:
            return
        
        print(f"Loading model from {self.config.checkpoint_path}...")
        try:
            checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')
            
            # Create pipeline
            self.pipeline = create_streaming_pipeline()
            self.pipeline.load_state_dict(checkpoint['model_state_dict'])
            self.pipeline.eval()
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Falling back to dummy mode...")
            self.use_model = False
            self.dummy_mode = True
    
    def initialize_face_aligner(self):
        """Initialize MediaPipe face aligner."""
        print("Initializing face aligner...")
        self.face_aligner = FaceAligner(running_mode='VIDEO')
        print("Face aligner ready!")
    
    def get_dummy_prediction(self) -> Dict:
        """Generate dummy prediction for UI testing."""
        # Cycle through emotions for demo
        self.dummy_pred_idx = (self.dummy_pred_idx + 1) % 7
        
        # Random-ish confidence
        self.dummy_confidence = 0.3 + 0.4 * np.random.random()
        
        probs = np.random.dirichlet(np.ones(7))
        probs[self.dummy_pred_idx] = self.dummy_confidence
        probs = probs / probs.sum()
        
        return {
            'pred_class': self.dummy_pred_idx,
            'confidence': self.dummy_confidence,
            'probs': probs,
            'embedding': np.random.randn(256).astype(np.float32),
            'mask': np.random.rand(224, 224).astype(np.float32),
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, Optional[FaceAlignmentResult]]:
        """Process single frame through pipeline."""
        self.frame_count += 1
        
        # Skip frames if configured
        if self.frame_count % self.config.process_every_n_frames != 0:
            return None, None
        
        # Face alignment
        if self.face_aligner is not None:
            align_result = self.face_aligner.align(frame)
        else:
            # Dummy alignment result
            h, w = frame.shape[:2]
            align_result = FaceAlignmentResult(
                success=True,
                aligned_face=cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0,
                landmarks_224=None,
                bbox=(w//4, h//4, w//2, h//2),
                error=None
            )
        
        if not align_result.success:
            return {'no_face': True}, align_result
        
        # Model inference or dummy
        if self.dummy_mode:
            pred = self.get_dummy_prediction()
        else:
            # Real model inference
            with torch.no_grad():
                # Prepare tensors
                aligned = align_result.aligned_face
                tensor = torch.from_numpy(aligned).permute(2, 0, 1).unsqueeze(0).float()
                
                # Normalize
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
                
                pred = self.pipeline.step(tensor, landmarks, visibility)
                pred = {k: v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v 
                       for k, v in pred.items()}
        
        return pred, align_result
    
    def draw_overlay(self, frame: np.ndarray, pred: Dict, 
                     align_result: Optional[FaceAlignmentResult]):
        """Draw all overlays on frame."""
        if pred is None:
            return
        
        if pred.get('no_face', False):
            self.visualizer.draw_no_face(frame)
            return
        
        bbox = align_result.bbox if align_result else None
        
        # Draw bbox
        if bbox:
            self.visualizer.draw_bbox(frame, bbox)
        
        # Draw emotion
        pred_class = pred.get('pred_class', 0)
        confidence = pred.get('confidence', 0.0)
        probs = pred.get('probs', None)
        
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        
        self.visualizer.draw_emotion_text(frame, bbox, pred_class, confidence, probs)
        
        # Draw landmarks
        if align_result and align_result.landmarks_224 is not None:
            # Project landmarks back to original frame coordinates
            # For simplicity, just draw on aligned face corner
            pass
        
        # Draw mask heatmap
        if 'mask' in pred and pred['mask'] is not None:
            mask = pred['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            self.visualizer.draw_mask_heatmap(frame, mask, bbox)
    
    def run_webcam(self):
        """Run demo on webcam."""
        cap = cv2.VideoCapture(self.config.webcam_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam {self.config.webcam_index}")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting webcam demo...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'f' - Toggle FPS display")
        print("  'b' - Toggle bbox display")
        print("  'c' - Toggle confidence display")
        print("  'm' - Toggle mask heatmap")
        print("  'l' - Toggle landmarks")
        print("  'd' - Toggle dummy/real mode")
        
        self.running = True
        prev_pred = None
        
        while self.running:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame
            pred, align_result = self.process_frame(frame)
            
            # Use previous prediction for smooth display
            if pred is not None and not pred.get('no_face', False):
                prev_pred = pred
            
            # Draw overlay
            if pred is not None:
                self.draw_overlay(frame, pred, align_result)
            elif prev_pred is not None:
                self.draw_overlay(frame, prev_pred, align_result)
            
            # Calculate FPS
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Draw FPS
            self.visualizer.draw_fps(frame, self.current_fps)
            
            # Draw mode indicator
            mode_text = "DUMMY MODE" if self.dummy_mode else "MODEL MODE"
            mode_color = (0, 165, 255) if self.dummy_mode else (0, 255, 0)
            self.visualizer.draw_status(frame, mode_text, mode_color)
            
            # Show frame
            cv2.imshow(self.config.window_name, frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"ferd_demo_{int(time.time())}.jpg", frame)
                print("Screenshot saved!")
            elif key == ord('f'):
                self.config.show_fps = not self.config.show_fps
            elif key == ord('b'):
                self.config.show_bbox = not self.config.show_bbox
            elif key == ord('c'):
                self.config.show_confidence = not self.config.show_confidence
            elif key == ord('m'):
                self.config.show_mask = not self.config.show_mask
            elif key == ord('l'):
                self.config.show_landmarks = not self.config.show_landmarks
            elif key == ord('d'):
                self.dummy_mode = not self.dummy_mode
                print(f"Switched to {'dummy' if self.dummy_mode else 'model'} mode")
            
            # Limit FPS
            frame_time = time.time() - loop_start
            target_time = 1.0 / self.config.max_fps
            if frame_time < target_time:
                time.sleep(target_time - frame_time)
        
        cap.release()
        cv2.destroyAllWindows()
        print("Demo stopped.")
    
    def run_video(self):
        """Run demo on video file."""
        if not self.config.video_path:
            print("No video path specified")
            return
        
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.config.video_path}")
            return
        
        print(f"Playing video: {self.config.video_path}")
        print("Press 'q' to quit, 'p' to pause, ' ' to step frame")
        
        self.running = True
        paused = False
        prev_pred = None
        
        while self.running:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break
                
                pred, align_result = self.process_frame(frame)
                
                if pred is not None and not pred.get('no_face', False):
                    prev_pred = pred
                
                if pred is not None:
                    self.draw_overlay(frame, pred, align_result)
                elif prev_pred is not None:
                    self.draw_overlay(frame, prev_pred, align_result)
                
                self.visualizer.draw_fps(frame, self.current_fps)
                mode_text = "DUMMY MODE" if self.dummy_mode else "MODEL MODE"
                self.visualizer.draw_status(frame, mode_text)
            
            cv2.imshow(self.config.window_name, frame)
            
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p') or key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                cv2.imwrite(f"ferd_demo_{int(time.time())}.jpg", frame)
                print("Screenshot saved!")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_image(self):
        """Run demo on single image."""
        if not self.config.image_path:
            print("No image path specified")
            return
        
        frame = cv2.imread(self.config.image_path)
        if frame is None:
            print(f"Error: Could not read image {self.config.image_path}")
            return
        
        pred, align_result = self.process_frame(frame)
        
        if pred is not None:
            self.draw_overlay(frame, pred, align_result)
        
        mode_text = "DUMMY MODE" if self.dummy_mode else "MODEL MODE"
        self.visualizer.draw_status(frame, mode_text)
        
        cv2.imshow(self.config.window_name, frame)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run(self):
        """Main entry point - routes to appropriate source handler."""
        self.initialize_face_aligner()
        self.initialize_model()
        
        try:
            if self.config.source == "webcam":
                self.run_webcam()
            elif self.config.source == "video":
                self.run_video()
            elif self.config.source == "image":
                self.run_image()
            else:
                print(f"Unknown source: {self.config.source}")
        finally:
            if self.face_aligner:
                self.face_aligner.close()
            cv2.destroyAllWindows()


def create_demo_config(
    source: str = "webcam",
    **kwargs
) -> DemoConfig:
    """Factory for demo configuration."""
    return DemoConfig(source=source, **kwargs)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="FERD Emotion Recognition Demo")
    parser.add_argument('--source', choices=['webcam', 'video', 'image'], default='webcam')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--image', type=str, help='Image file path')
    parser.add_argument('--webcam', type=int, default=0, help='Webcam index')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--model', action='store_true', help='Use trained model')
    parser.add_argument('--dummy', action='store_true', help='Force dummy mode')
    parser.add_argument('--no-bbox', action='store_true', help='Hide bounding box')
    parser.add_argument('--no-conf', action='store_true', help='Hide confidence')
    parser.add_argument('--show-mask', action='store_true', help='Show SAFM mask heatmap')
    parser.add_argument('--show-landmarks', action='store_true', help='Show landmarks')
    
    args = parser.parse_args()
    
    config = create_demo_config(
        source=args.source,
        video_path=args.video or "",
        image_path=args.image or "",
        webcam_index=args.webcam,
        checkpoint_path=args.checkpoint or "",
        use_trained_model=args.model and not args.dummy,
        show_bbox=not args.no_bbox,
        show_confidence=not args.no_conf,
        show_mask=args.show_mask,
        show_landmarks=args.show_landmarks,
    )
    
    harness = DemoHarness(config)
    harness.run()


if __name__ == "__main__":
    # Quick test with dummy mode
    if len(sys.argv) == 1:
        print("Running quick dummy demo test...")
        config = create_demo_config(source="webcam")
        harness = DemoHarness(config)
        harness.initialize_face_aligner()
        
        # Test one frame
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            pred, align = harness.process_frame(frame)
            harness.draw_overlay(frame, pred, align)
            
            cv2.imshow("Test", frame)
            print("Press any key to close test window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Demo harness test passed!")
        else:
            print("No webcam available for test")
    else:
        main()