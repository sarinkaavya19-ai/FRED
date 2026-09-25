"""
Region Confidence Scoring for Heuristic SAFM (Spatial-Attention Facial Masking)
Computes per-region visibility from MediaPipe landmark confidence + local contrast.

SPRINT-SIMPLIFICATION: Non-learned heuristic version.
Full PMD: Trainable gating network (safm/attention_mask.py) with learned attention.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FaceRegion(Enum):
    """Facial regions for SAFM masking (MediaPipe landmark indices)."""
    # Using MediaPipe FaceLandmarker 478 landmark indices
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_BROW = "left_brow"
    RIGHT_BROW = "right_brow"
    NOSE = "nose"
    MOUTH = "mouth"
    JAW = "jaw"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    FOREHEAD = "forehead"


# MediaPipe landmark indices for each region
# Based on MediaPipe Face Mesh topology (468 + 10 iris = 478)
REGION_LANDMARKS = {
    FaceRegion.LEFT_EYE: [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        468, 469, 470, 471, 472  # iris landmarks
    ],
    FaceRegion.RIGHT_EYE: [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
        473, 474, 475, 476, 477  # iris landmarks
    ],
    FaceRegion.LEFT_BROW: [
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46
    ],
    FaceRegion.RIGHT_BROW: [
        336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    ],
    FaceRegion.NOSE: [
        1, 2, 5, 4, 6, 19, 20, 94, 125, 142, 36, 220, 115, 48, 64, 98, 97, 2, 326, 327,
        298, 299, 330, 279, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290,
        291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306
    ],
    FaceRegion.MOUTH: [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14,
        87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415
    ],
    FaceRegion.JAW: [
        172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323,
        454, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162
    ],
    FaceRegion.LEFT_CHEEK: [
        117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187, 207, 216
    ],
    FaceRegion.RIGHT_CHEEK: [
        346, 347, 348, 349, 350, 355, 436, 280, 415, 416, 417, 423, 385, 376, 375, 374, 373
    ],
    FaceRegion.FOREHEAD: [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377
    ],
}


@dataclass
class RegionConfidence:
    """Confidence score for a facial region."""
    region: FaceRegion
    landmark_confidence: float  # Average landmark visibility/presence
    contrast_score: float       # Local contrast (brightness variance)
    combined_score: float       # Weighted combination
    landmark_indices: List[int] # Indices used


class RegionConfidenceScorer:
    """
    Computes per-region confidence scores for SAFM.
    
    Two signals:
    1. Landmark visibility: MediaPipe provides per-landmark presence confidence
    2. Local contrast: Brightness variance in region (low contrast = poor lighting/occlusion)
    """
    
    def __init__(
        self,
        landmark_weight: float = 0.6,
        contrast_weight: float = 0.4,
        contrast_window: int = 15,
        min_contrast: float = 0.01,
        max_contrast: float = 0.5,
    ):
        """
        Args:
            landmark_weight: Weight for landmark visibility score
            contrast_weight: Weight for local contrast score
            contrast_window: Window size for local contrast computation
            min_contrast: Minimum contrast for normalization
            max_contrast: Maximum contrast for normalization
        """
        self.landmark_weight = landmark_weight
        self.contrast_weight = contrast_weight
        self.contrast_window = contrast_window
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        
        # Normalize weights
        total = landmark_weight + contrast_weight
        self.landmark_weight = landmark_weight / total
        self.contrast_weight = contrast_weight / total
    
    def compute_landmark_confidence(
        self,
        landmarks: np.ndarray,        # (478, 3) - x, y, z/visibility
        visibility: Optional[np.ndarray] = None,  # (478,) - MediaPipe visibility
        presence: Optional[np.ndarray] = None,    # (478,) - MediaPipe presence
    ) -> Dict[FaceRegion, float]:
        """
        Compute average landmark confidence per region.
        
        MediaPipe FaceLandmarker provides:
        - landmark.x, landmark.y, landmark.z (normalized)
        - landmark.visibility (0-1, how visible)
        - landmark.presence (0-1, how confidently present)
        
        Args:
            landmarks: (N, 3) array of x, y, z
            visibility: (N,) visibility scores
            presence: (N,) presence scores
            
        Returns:
            Dict mapping region -> average confidence (0-1)
        """
        # Use visibility if available, else presence, else assume 1.0
        if visibility is not None:
            conf = visibility
        elif presence is not None:
            conf = presence
        else:
            # Infer from z-coordinate (negative = behind camera)
            conf = np.clip(-landmarks[:, 2] + 0.5, 0, 1)
        
        region_scores = {}
        for region, indices in REGION_LANDMARKS.items():
            # Filter valid indices
            valid_indices = [i for i in indices if i < len(conf)]
            if valid_indices:
                region_scores[region] = float(np.mean(conf[valid_indices]))
            else:
                region_scores[region] = 0.0
        
        return region_scores
    
    def compute_local_contrast(
        self,
        image: np.ndarray,           # (H, W, 3) RGB, float32 [0,1] or uint8
        landmarks: np.ndarray,       # (478, 2) pixel coordinates
    ) -> Dict[FaceRegion, float]:
        """
        Compute local contrast (brightness variance) per region.
        
        Low contrast indicates:
        - Poor lighting
        - Occlusion (hand, mask)
        - Blur/motion
        
        Args:
            image: Aligned face image (224, 224, 3)
            landmarks: Landmark pixel coordinates in image space
            
        Returns:
            Dict mapping region -> normalized contrast score (0-1)
        """
        # Convert to grayscale for contrast computation
        if image.dtype != np.uint8:
            gray = (image * 255).astype(np.uint8)
        else:
            gray = image
        
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        
        h, w = gray.shape
        region_contrasts = {}
        
        for region, indices in REGION_LANDMARKS.items():
            # Get valid landmark coordinates for this region
            valid_pts = []
            for idx in indices:
                if idx < len(landmarks):
                    x, y = landmarks[idx]
                    if 0 <= x < w and 0 <= y < h:
                        valid_pts.append((int(x), int(y)))
            
            if not valid_pts:
                region_contrasts[region] = 0.0
                continue
            
            # Compute bounding box of region landmarks
            xs = [p[0] for p in valid_pts]
            ys = [p[1] for p in valid_pts]
            x1, x2 = max(0, min(xs) - self.contrast_window), min(w, max(xs) + self.contrast_window)
            y1, y2 = max(0, min(ys) - self.contrast_window), min(h, max(ys) + self.contrast_window)
            
            if x2 <= x1 or y2 <= y1:
                region_contrasts[region] = 0.0
                continue
            
            # Extract region patch
            patch = gray[y1:y2, x1:x2]
            
            if patch.size == 0:
                region_contrasts[region] = 0.0
                continue
            
            # Compute local contrast as standard deviation / mean (coefficient of variation)
            # Normalize to [0, 1]
            patch_mean = patch.mean()
            patch_std = patch.std()
            
            if patch_mean > 0:
                contrast = patch_std / patch_mean
            else:
                contrast = 0.0
            
            # Normalize contrast to [0, 1] range
            contrast_norm = np.clip(
                (contrast - self.min_contrast) / (self.max_contrast - self.min_contrast),
                0, 1
            )
            
            region_contrasts[region] = float(contrast_norm)
        
        return region_contrasts
    
    def compute_region_confidences(
        self,
        image: np.ndarray,                    # (H, W, 3) aligned face
        landmarks: np.ndarray,                # (478, 2) or (478, 3) pixel coords
        visibility: Optional[np.ndarray] = None,  # (478,) MediaPipe visibility
        presence: Optional[np.ndarray] = None,    # (478,) MediaPipe presence
    ) -> Dict[FaceRegion, RegionConfidence]:
        """
        Compute combined confidence scores for all regions.
        
        Returns:
            Dict[FaceRegion, RegionConfidence] with all scores
        """
        # Ensure landmarks are 2D for contrast computation
        if landmarks.shape[1] >= 3:
            landmarks_2d = landmarks[:, :2]
            landmarks_3d = landmarks
        else:
            landmarks_2d = landmarks
            landmarks_3d = np.hstack([landmarks, np.zeros((len(landmarks), 1))])
        
        # Compute landmark confidence
        landmark_scores = self.compute_landmark_confidence(
            landmarks_3d, visibility, presence
        )
        
        # Compute contrast scores
        contrast_scores = self.compute_local_contrast(image, landmarks_2d)
        
        # Combine
        results = {}
        for region in FaceRegion:
            lm_score = landmark_scores.get(region, 0.0)
            ct_score = contrast_scores.get(region, 0.0)
            
            combined = (
                self.landmark_weight * lm_score +
                self.contrast_weight * ct_score
            )
            
            # Get landmark indices for this region
            indices = REGION_LANDMARKS.get(region, [])
            valid_indices = [i for i in indices if i < len(landmarks_3d)]
            
            results[region] = RegionConfidence(
                region=region,
                landmark_confidence=lm_score,
                contrast_score=ct_score,
                combined_score=combined,
                landmark_indices=valid_indices,
            )
        
        return results


def create_region_scorer(**kwargs) -> RegionConfidenceScorer:
    """Factory function."""
    return RegionConfidenceScorer(**kwargs)


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing RegionConfidenceScorer...")
    
    scorer = create_region_scorer()
    
    # Dummy aligned face (224x224)
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Dummy landmarks (478, 2) - roughly centered
    dummy_landmarks = np.zeros((478, 2), dtype=np.float32)
    for i in range(478):
        # Simple face-like distribution
        angle = (i / 478) * 2 * np.pi
        r = 80 + 20 * np.sin(angle * 3)
        dummy_landmarks[i] = [
            112 + r * np.cos(angle),
            112 + r * np.sin(angle)
        ]
    
    # Dummy visibility (all visible)
    dummy_visibility = np.ones(478, dtype=np.float32)
    
    # Compute confidences
    confidences = scorer.compute_region_confidences(
        dummy_img, dummy_landmarks, visibility=dummy_visibility
    )
    
    print("\nRegion Confidences:")
    for region, conf in confidences.items():
        print(f"  {region.value:15s}: landmark={conf.landmark_confidence:.3f}, "
              f"contrast={conf.contrast_score:.3f}, combined={conf.combined_score:.3f}")
    
    print("\n[OK] Region confidence scoring test passed!")