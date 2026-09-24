"""
Face Detection & Alignment Module for FERD Sprint
Uses MediaPipe FaceLandmarker (BlazeFace-based) for detection and 468-point landmarks.
Outputs aligned, cropped face tensor at 224x224 resolution.

SPRINT-SIMPLIFICATION: Uses MediaPipe FaceLandmarker as-is per PMD Section 7.
Full PMD: Could swap to BlazeFace directly for <2ms latency on edge NPU.
MediaPipe 1.0+ uses tasks.vision.FaceLandmarker API.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass
import time
import os


@dataclass
class FaceAlignmentResult:
    """Result of face alignment operation."""
    success: bool
    aligned_face: Optional[np.ndarray] = None      # 224x224x3 RGB, float32 [0,1]
    landmarks_224: Optional[np.ndarray] = None     # (468, 2) landmarks in 224x224 space
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in original frame
    error: Optional[str] = None


class FaceAligner:
    """
    MediaPipe FaceLandmarker based face detection and alignment.
    
    Uses 468 landmarks to compute affine transform for pose normalization.
    Key landmarks for alignment: left eye (33), right eye (263), nose tip (1), mouth center (13/14).
    """
    
    # MediaPipe FaceLandmarker landmark indices for alignment
    LEFT_EYE_IDX = 33       # Left eye center
    RIGHT_EYE_IDX = 263     # Right eye center
    NOSE_TIP_IDX = 1        # Nose tip
    MOUTH_LEFT_IDX = 61     # Mouth left corner
    MOUTH_RIGHT_IDX = 291   # Mouth right corner
    
    def __init__(
        self,
        output_size: int = 224,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        running_mode: str = "VIDEO",  # "IMAGE" or "VIDEO" or "LIVE_STREAM"
        model_asset_path: Optional[str] = None,
    ):
        """
        Initialize MediaPipe FaceLandmarker.
        
        Args:
            output_size: Output crop size (default 224x224 per PMD)
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            max_num_faces: Maximum faces to detect (1 for single-subject demo)
            running_mode: "IMAGE" for static, "VIDEO" for tracking, "LIVE_STREAM" for async
            model_asset_path: Path to face_landmarker.task model (auto-download if None)
        """
        self.output_size = output_size
        self.running_mode = running_mode.upper()
        
        # Default model path - MediaPipe will auto-download if not provided
        if model_asset_path is None:
            # Try to find in mediapipe cache or use default
            model_asset_path = self._get_default_model_path()
        
        # Configure FaceLandmarker
        base_options = mp_python.BaseOptions(model_asset_path=model_asset_path)
        
        if self.running_mode == "VIDEO":
            running_mode_enum = vision.RunningMode.VIDEO
        elif self.running_mode == "LIVE_STREAM":
            running_mode_enum = vision.RunningMode.LIVE_STREAM
        else:
            running_mode_enum = vision.RunningMode.IMAGE
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode_enum,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Target landmark positions in output space (224x224)
        self.target_landmarks = self._get_target_landmarks(output_size)
        
        # Source landmark indices used for alignment transform
        self.src_indices = np.array([
            self.LEFT_EYE_IDX,      # Left eye
            self.RIGHT_EYE_IDX,     # Right eye
            self.NOSE_TIP_IDX,      # Nose tip
            self.MOUTH_LEFT_IDX,    # Mouth left
            self.MOUTH_RIGHT_IDX,   # Mouth right
        ], dtype=np.int32)
        
        self.target_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        
        # Timestamp for VIDEO mode (monotonically increasing)
        self._timestamp_ms = 0
    
    def _get_default_model_path(self) -> str:
        """Get default MediaPipe face landmarker model path."""
        # MediaPipe 1.0 requires explicit model file path
        # Auto-download to cache if not present
        import os
        cache_dir = os.path.expanduser('~/.cache/mediapipe')
        model_path = os.path.join(cache_dir, 'face_landmarker.task')
        
        if not os.path.exists(model_path):
            # Try to download
            try:
                import urllib.request
                os.makedirs(cache_dir, exist_ok=True)
                url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
                print(f'Downloading MediaPipe face landmarker model to {model_path}...')
                urllib.request.urlretrieve(url, model_path)
                print('Download complete.')
            except Exception as e:
                raise RuntimeError(f'Failed to download face_landmarker.task: {e}')
        
        return model_path
    
    def _get_target_landmarks(self, size: int) -> np.ndarray:
        """
        Define canonical target landmark positions for aligned face.
        These are the "ideal" positions in the 224x224 output space.
        """
        # Normalized coordinates (0-1) for canonical face
        targets_norm = np.array([
            [0.35, 0.35],   # Left eye center
            [0.65, 0.35],   # Right eye center
            [0.50, 0.55],   # Nose tip
            [0.40, 0.75],   # Mouth left corner
            [0.60, 0.75],   # Mouth right corner
        ], dtype=np.float32)
        
        return targets_norm * size
    
    def _estimate_affine_transform(
        self, 
        src_pts: np.ndarray, 
        dst_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate affine transformation matrix from src to dst points.
        Uses OpenCV's estimateAffinePartial2D (handles overdetermined systems).
        """
        if len(src_pts) < 3:
            return None
        
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts.reshape(-1, 1, 2).astype(np.float32),
            dst_pts.reshape(-1, 1, 2).astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        return M
    
    def _landmarks_to_array(self, landmarks, image_shape) -> np.ndarray:
        """Convert MediaPipe normalized landmarks to pixel coordinates."""
        h, w = image_shape[:2]
        pts = np.array([
            [lm.x * w, lm.y * h] for lm in landmarks
        ], dtype=np.float32)
        return pts
    
    def align(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> FaceAlignmentResult:
        """
        Detect, align, and crop face from frame.
        
        Args:
            frame: BGR image (H, W, 3) from OpenCV
            timestamp_ms: Required for VIDEO mode (monotonically increasing)
            
        Returns:
            FaceAlignmentResult with aligned face tensor or error info
        """
        start_time = time.perf_counter()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process with FaceLandmarker
        if self.running_mode == "VIDEO":
            if timestamp_ms is None:
                timestamp_ms = self._timestamp_ms
            self._timestamp_ms = timestamp_ms + 33  # ~30 FPS increment
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            results = self.landmarker.detect(mp_image)
        
        if not results.face_landmarks:
            return FaceAlignmentResult(
                success=False,
                error="no_face_detected"
            )
        
        # Take first (most confident) face
        face_landmarks = results.face_landmarks[0]
        
        # Convert landmarks to pixel coordinates
        landmarks_px = self._landmarks_to_array(face_landmarks, rgb_frame.shape)
        
        # Extract source points for alignment
        src_pts = landmarks_px[self.src_indices].astype(np.float32)
        dst_pts = self.target_landmarks[self.target_indices].astype(np.float32)
        
        # Estimate affine transform
        M = self._estimate_affine_transform(src_pts, dst_pts)
        
        if M is None:
            return FaceAlignmentResult(
                success=False,
                error="alignment_failed"
            )
        
        # Warp the full frame to aligned space
        aligned_rgb = cv2.warpAffine(
            rgb_frame,
            M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        
        # Transform all landmarks to aligned space
        ones = np.ones((landmarks_px.shape[0], 1), dtype=np.float32)
        landmarks_h = np.hstack([landmarks_px, ones])
        landmarks_aligned = (M @ landmarks_h.T).T  # (468, 2)
        
        # Compute bounding box in original frame
        x_coords = landmarks_px[:, 0]
        y_coords = landmarks_px[:, 1]
        x1, x2 = int(x_coords.min()), int(x_coords.max())
        y1, y2 = int(y_coords.min()), int(y_coords.max())
        bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Normalize to [0, 1] float32 for model input
        aligned_tensor = aligned_rgb.astype(np.float32) / 255.0
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return FaceAlignmentResult(
            success=True,
            aligned_face=aligned_tensor,      # (224, 224, 3) float32 [0,1]
            landmarks_224=landmarks_aligned,  # (468, 2) in 224x224 space
            bbox=bbox,
            error=None,
        )
    
    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()


def create_face_aligner(**kwargs) -> FaceAligner:
    """Factory function for easy instantiation."""
    return FaceAligner(**kwargs)


# Convenience function for single-frame use (e.g., testing)
def align_face(
    frame: np.ndarray,
    output_size: int = 224,
    **kwargs
) -> FaceAlignmentResult:
    """
    One-shot face alignment for testing.
    Creates aligner, processes frame, closes aligner.
    
    For video streams, use FaceAligner class directly to reuse tracking.
    """
    aligner = FaceAligner(output_size=output_size, running_mode="IMAGE", **kwargs)
    try:
        return aligner.align(frame)
    finally:
        aligner.close()


if __name__ == "__main__":
    # Quick self-test with webcam
    import sys
    
    print("Testing FaceAligner with webcam...")
    print("Press 'q' to quit, 's' to save aligned face")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        sys.exit(1)
    
    aligner = FaceAligner(running_mode="VIDEO")
    
    try:
        timestamp = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Align face
            result = aligner.align(frame, timestamp_ms=timestamp)
            timestamp += 33
            
            # Visualize
            vis = frame.copy()
            
            if result.success:
                # Draw bbox
                x, y, w, h = result.bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Show aligned face in corner
                aligned_vis = (result.aligned_face * 255).astype(np.uint8)
                aligned_vis = cv2.cvtColor(aligned_vis, cv2.COLOR_RGB2BGR)
                h_a, w_a = aligned_vis.shape[:2]
                vis[10:10+h_a, 10:10+w_a] = aligned_vis
                
                cv2.putText(vis, "Face Detected", (10, 230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, f"No Face: {result.error}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Face Alignment Test", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and result.success:
                cv2.imwrite("aligned_face_test.jpg", 
                           (result.aligned_face * 255).astype(np.uint8)[:, :, ::-1])
                print("Saved aligned_face_test.jpg")
    
    finally:
        aligner.close()
        cap.release()
        cv2.destroyAllWindows()