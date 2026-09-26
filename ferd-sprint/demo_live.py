"""
FERD Live Demo - Landmark-Based Emotion Detection
Real-time emotion detection from MediaPipe 478 facial landmarks.
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import deque
from typing import Optional, Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.face_align import FaceAligner, FaceAlignmentResult

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
EMOTION_ICONS = [">:(", "D:<", "O_O", ":)", ":(", "O_O", "|_|"]

EMOTION_COLORS = [
    (0, 0, 255),      # Anger - Red
    (0, 128, 0),      # Disgust - Dark Green
    (128, 0, 128),    # Fear - Purple
    (0, 255, 255),    # Happiness - Yellow
    (255, 0, 0),      # Sadness - Blue
    (0, 165, 255),    # Surprise - Orange
    (128, 128, 128),  # Neutral - Gray
]


class LandmarkEmotionDetector:
    """Detects emotions from MediaPipe 478 facial landmarks using geometric features."""
    
    # MediaPipe Face Mesh landmark indices
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263
    
    LEFT_EYEBROW_INNER = 70
    LEFT_EYEBROW_TOP = 66
    RIGHT_EYEBROW_INNER = 336
    RIGHT_EYEBROW_TOP = 296
    
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    
    def __init__(self):
        self.history = deque(maxlen=5)
        
    def _dist(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))
    
    def _mouth_smile(self, lm: np.ndarray) -> float:
        """Positive = smile, Negative = frown"""
        left = lm[self.MOUTH_LEFT]
        right = lm[self.MOUTH_RIGHT]
        top = lm[self.MOUTH_TOP]
        bottom = lm[self.MOUTH_BOTTOM]
        
        center_y = (top[1] + bottom[1]) / 2
        left_diff = left[1] - center_y
        right_diff = right[1] - center_y
        avg_diff = (left_diff + right_diff) / 2
        
        mouth_width = self._dist(left, right)
        return -avg_diff / (mouth_width + 1e-6)
    
    def _mouth_open(self, lm: np.ndarray) -> float:
        top = lm[self.MOUTH_TOP]
        bottom = lm[self.MOUTH_BOTTOM]
        left = lm[self.MOUTH_LEFT]
        right = lm[self.MOUTH_RIGHT]
        
        h = self._dist(top, bottom)
        w = self._dist(left, right)
        return h / (w + 1e-6)
    
    def _eye_open(self, lm: np.ndarray, left: bool) -> float:
        if left:
            t, b, l, r = self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, self.LEFT_EYE_LEFT, self.LEFT_EYE_RIGHT
        else:
            t, b, l, r = self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM, self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT
        
        h = self._dist(lm[t], lm[b])
        w = self._dist(lm[l], lm[r])
        return h / (w + 1e-6)
    
    def _brow_height(self, lm: np.ndarray, left: bool) -> float:
        """Positive = raised eyebrows"""
        if left:
            brow = lm[self.LEFT_EYEBROW_TOP]
            eye = lm[self.LEFT_EYE_TOP]
        else:
            brow = lm[self.RIGHT_EYEBROW_TOP]
            eye = lm[self.RIGHT_EYE_TOP]
        
        return (eye[1] - brow[1]) / 100.0
    
    def _brow_furrow(self, lm: np.ndarray) -> float:
        """Inner brows distance. Higher = more furrowed (brows closer together)."""
        left = lm[self.LEFT_EYEBROW_INNER]
        right = lm[self.RIGHT_EYEBROW_INNER]
        dist = self._dist(left, right)
        return max(0.0, 1.0 - dist / 80.0)  # 0 to 1 scale
    
    # ANGER: furrowed brows + tight eyes + frown (highest priority)
        if furrow > 0.3 and eye_open < 0.25:
            scores[0] = min(0.98, 0.5 + furrow * 3.0 + (0.25 - eye_open) * 3.0 + max(0, -smile) * 2.0)
        
        # HAPPINESS: smile (priority 2)
        if smile > 0.015:
            scores[3] = min(0.98, 0.7 + smile * 10)
        
        # SURPRISE: VERY wide eyes + HIGH raised brows + open mouth (strict thresholds)
        if eye_open > 0.45 and brow > 0.35 and mouth_open > 0.5:
            scores[5] = min(0.98, 0.5 + eye_open * 1.5 + brow * 1.5 + mouth_open * 0.5)
        
        # FEAR: wide eyes + raised brows + tense mouth (no smile)
        if eye_open > 0.35 and brow > 0.2 and smile < 0.02:
            scores[2] = min(0.9, 0.2 + eye_open * 1.5 + brow * 1.2)
        
        # ANGER: stronger check
        if furrow > 0.5 and eye_open < 0.22:
            scores[0] = max(scores[0], min(0.98, 0.5 + furrow * 3.0 + (0.25 - eye_open) * 3.0 + max(0, -smile) * 2.0))
        
        # SADNESS: frown + droopy eyes
        if smile < -0.01:
            scores[4] = min(0.95, 0.3 - smile * 4 + (0.25 - eye_open) * 1.5)
        
        # DISGUST: frown + narrowed eyes
        if smile < -0.005 and eye_open < 0.22 and eye_open > 0.12:
            scores[1] = min(0.75, 0.25 - smile * 2)
        
        # ANGER: stronger check
        if furrow > 0.6 and eye_open < 0.2:
            scores[0] = max(scores[0], min(0.98, 0.6 + furrow * 4.0 + (0.2 - eye_open) * 4.0 + max(0, -smile) * 2.5))
        
        # NEUTRAL fallback
        max_other = scores.max()
        if max_other < 0.3:
            scores[6] = 0.98 - max_other
        
        # Normalize
        scores = np.clip(scores, 0, 1)
        total = scores.sum()
        if total > 0:
            scores = scores / total
        else:
            scores[6] = 1.0
        
        # Temporal smoothing with adaptive weighting
        self.history.append(scores)
        
        # Adaptive smoothing: if top emotion changes significantly, reset history
        if len(self.history) > 1:
            prev_top = np.argmax(self.history[-2])
            curr_top = np.argmax(scores)
            if prev_top != curr_top:
                # Emotion changed significantly - clear history for instant adaptation
                self.history.clear()
                self.history.append(scores)
                smoothed = scores
            else:
                # Same emotion - normal smoothing
                weights = np.ones(len(self.history))
                weights = weights / weights.sum()
                smoothed = np.average(self.history, axis=0, weights=weights)
        else:
            smoothed = scores
        
        smoothed = smoothed / smoothed.sum()
        
        return {EMOTIONS[i]: float(smoothed[i]) for i in range(7)}


class LiveEmotionDemo:
    """Live webcam demo with landmark-based emotion detection."""
    
    def __init__(self):
        self.face_aligner = None
        self.emotion_detector = None
        self.conf_history = deque(maxlen=100)
        
    def initialize(self):
        print("[INIT] Initializing Face Aligner...")
        self.face_aligner = FaceAligner(running_mode='VIDEO')
        self.emotion_detector = LandmarkEmotionDetector()
        print("[OK] Face Aligner ready (MediaPipe 478 landmarks)")
        print("[OK] Landmark Emotion Detector ready")
        
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        if self.face_aligner is None:
            self.face_aligner = FaceAligner(running_mode='VIDEO')
            self.emotion_detector = LandmarkEmotionDetector()
        
        align_result = self.face_aligner.align(frame)
        if not align_result.success or align_result.landmarks_224 is None:
            return None
        
        landmarks = align_result.landmarks_224[:, :2]
        emotions = self.emotion_detector.detect(landmarks)
        
        top_emo = max(emotions, key=emotions.get)
        confidence = emotions[top_emo]
        pred_idx = EMOTIONS.index(top_emo)
        
        return {
            'pred': EMOTIONS.index(top_emo),
            'conf': confidence,
            'probs': np.array([emotions[e] for e in EMOTIONS]),
            'bbox': align_result.bbox,
            'all_emotions': emotions
        }
    
    def draw_main(self, frame: np.ndarray, pred: int, conf: float, bbox: Optional[Tuple], x: int, y: int):
        emo = EMOTIONS[pred]
        icon = EMOTION_ICONS[pred]
        color = (0, 255, 0) if conf > 0.5 else (0, 200, 255) if conf > 0.3 else (0, 100, 255)
        cv2.putText(frame, f"[+] {icon} {emo}: {conf:.0%}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        if bbox:
            x1, y1, w, h = bbox
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
    
    def draw_bars(self, frame: np.ndarray, probs: Dict[str, float], x: int, y: int):
        max_p = max(probs.values())
        for i, (emo, prob) in enumerate(probs.items()):
            by = y + i * 28
            cv2.rectangle(frame, (x, by), (x + 300, by + 22), (40, 40, 40), -1)
            cv2.rectangle(frame, (x, by), (x + 300, by + 22), (80, 80, 80), 1)
            
            max_p = max(probs.values())
            fill = int(300 * prob / max_p) if max_p > 0 else 0
            fill = max(fill, 3) if prob > 0.01 else 0
            
            color = EMOTION_COLORS[EMOTIONS.index(list(probs.keys())[i])]
            for w in range(fill):
                fc = tuple(int(c * (0.4 + 0.6 * w / 300)) for c in EMOTION_COLORS[EMOTIONS.index(list(probs.keys())[i])])
                cv2.line(frame, (x + w, by), (x + w, by + 22), fc, 1)
            
            cv2.putText(frame, f"{EMOTION_ICONS[i]} {list(probs.keys())[i]}", (x - 90, by + 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{prob:.0%}", (x + int(300 * prob / max(probs.values())) + 8, by + 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No webcam found")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n[LIVE] Landmark Emotion Detection - Press 'q' to quit")
        
        self.face_aligner = FaceAligner(running_mode='VIDEO')
        self.emotion_detector = LandmarkEmotionDetector()
        
        last_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame)
            display = frame.copy()
            
            if result:
                self.draw_main(frame, result['pred'], result['conf'], result['bbox'], 10, 40)
                self.draw_bars(frame, result['all_emotions'], 10, 80)
            
            # FPS
            now = time.time()
            fps = 1 / (now - getattr(self, '_last_time', now))
            self._last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "LANDMARK EMOTION DETECTOR", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "q=quit", (10, frame.shape[0] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            
            cv2.imshow("FERD Live - Landmark Emotion Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n[BYE] Demo ended")


EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
EMOTION_ICONS = [">:(", "D:<", "O_O", ":)", ":(", "O_O", "|_|"]
EMOTION_COLORS = [(0, 0, 255), (0, 128, 0), (128, 0, 128), (0, 255, 255), (255, 0, 0), (0, 165, 255), (128, 128, 128)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    print("=" * 50)
    print("FERD Live - Landmark Emotion Detection")
    print("Real emotion detection from facial landmarks")
    print("=" * 50)
    
    demo = LiveEmotionDemo()
    try:
        demo.initialize()
        demo.run()
    except KeyboardInterrupt:
        print("\n[BYE]")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()