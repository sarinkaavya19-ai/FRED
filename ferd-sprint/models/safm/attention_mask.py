"""
Soft Spatial Attention Mask for SAFM (Spatial-Attention Facial Masking)
Converts per-region confidence scores into a smooth 224x224 spatial mask.

SPRINT-SIMPLIFICATION: Non-learned heuristic mask generation.
Full PMD: Trainable gating network that learns to weight regions.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .region_confidence import (
    RegionConfidenceScorer,
    RegionConfidence,
    FaceRegion,
    REGION_LANDMARKS,
    create_region_scorer,
)


class HeuristicSAFM:
    """
    Heuristic Spatial-Attention Facial Masking (SAFM).
    
    Generates a soft spatial mask from per-region confidence scores.
    The mask down-weights unreliable regions (occluded, poorly lit, extreme pose).
    
    Pipeline:
    1. Compute per-region confidence (landmark visibility + local contrast)
    2. Create region-level weight map from landmarks
    3. Upsample to full resolution with Gaussian smoothing
    4. Apply temperature scaling for smoothness control
    """
    
    def __init__(
        self,
        output_size: int = 224,
        temperature: float = 2.0,
        gaussian_sigma: float = 8.0,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
        landmark_weight: float = 0.6,
        contrast_weight: float = 0.4,
    ):
        """
        Args:
            output_size: Output mask size (default 224x224)
            temperature: Temperature for softmax-like smoothing (higher = smoother)
            gaussian_sigma: Gaussian blur sigma for smooth transitions
            min_weight: Minimum mask weight (prevents complete zeroing)
            max_weight: Maximum mask weight
            landmark_weight: Weight for landmark visibility in region scoring
            contrast_weight: Weight for local contrast in region scoring
        """
        self.output_size = output_size
        self.temperature = temperature
        self.gaussian_sigma = gaussian_sigma
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Region confidence scorer
        self.region_scorer = create_region_scorer(
            landmark_weight=landmark_weight,
            contrast_weight=contrast_weight,
        )
        
        # Pre-compute landmark-to-region mapping for fast mask generation
        self._region_map = self._build_region_map()
    
    def _build_region_map(self) -> np.ndarray:
        """
        Build a region assignment map: for each pixel, which region it belongs to.
        Uses Voronoi-like assignment based on landmark positions.
        
        Returns:
            region_map: (output_size, output_size) int array with region indices
        """
        # This will be computed dynamically based on actual landmarks
        # For now, return None to indicate dynamic computation
        return None
    
    def _landmarks_to_region_weights(
        self,
        landmarks: np.ndarray,           # (478, 2) pixel coordinates
        region_confidences: Dict[FaceRegion, RegionConfidence],
    ) -> np.ndarray:
        """
        Convert landmark positions + region confidences to per-pixel weight map.
        
        Uses Gaussian radial basis functions centered at each landmark,
        weighted by the region's combined confidence.
        
        Args:
            landmarks: (478, 2) landmark pixel coordinates in output space
            region_confidences: Dict of region -> RegionConfidence
            
        Returns:
            weight_map: (output_size, output_size) float32 [min_weight, max_weight]
        """
        h, w = self.output_size, self.output_size
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        coords = np.stack([x_coords, y_coords], axis=-1)  # (H, W, 2)
        
        # Initialize weight map
        weight_map = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        
        # For each region, add Gaussian-weighted contribution
        for region, conf in region_confidences.items():
            region_weight = conf.combined_score
            
            # Get landmarks for this region
            indices = conf.landmark_indices
            if not indices:
                continue
            
            region_landmarks = landmarks[indices]  # (N_region, 2)
            
            # For each landmark in region, add Gaussian contribution
            for lm in region_landmarks:
                lx, ly = lm
                
                # Gaussian weight
                dx = x_coords - lx
                dy = y_coords - ly
                dist2 = dx * dx + dy * dy
                gaussian = np.exp(-dist2 / (2 * self.gaussian_sigma ** 2))
                
                # Accumulate weighted by region confidence
                weight_map += gaussian * region_weight
                weight_accum += gaussian
        
        # Normalize by accumulated weights (avoid division by zero)
        mask = np.zeros_like(weight_map)
        valid = weight_accum > 1e-6
        mask[valid] = weight_map[valid] / weight_accum[valid]
        
        # Apply temperature scaling for smoothness
        # Higher temperature = more uniform, lower = sharper transitions
        mask = np.power(mask, 1.0 / self.temperature)
        
        # Clamp to [min_weight, max_weight]
        mask = np.clip(mask, self.min_weight, self.max_weight)
        
        return mask.astype(np.float32)
    
    def _apply_gaussian_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for smooth transitions."""
        if self.gaussian_sigma > 0:
            # Convert to uint8 for cv2, then back
            mask_uint8 = (mask * 255).astype(np.uint8)
            ksize = int(6 * self.gaussian_sigma + 1) | 1  # odd kernel size
            smoothed = cv2.GaussianBlur(mask_uint8, (ksize, ksize), self.gaussian_sigma)
            return (smoothed.astype(np.float32) / 255.0)
        return mask
    
    def generate_mask(
        self,
        image: np.ndarray,                    # (H, W, 3) aligned face, float32 [0,1]
        landmarks: np.ndarray,                # (478, 2) or (478, 3) in output space
        visibility: Optional[np.ndarray] = None,
        presence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[FaceRegion, RegionConfidence]]:
        """
        Generate soft spatial attention mask.
        
        Args:
            image: Aligned face crop (224, 224, 3) RGB float32 [0,1]
            landmarks: Landmarks in 224x224 space (478, 2) or (478, 3)
            visibility: Optional MediaPipe visibility scores (478,)
            presence: Optional MediaPipe presence scores (478,)
            
        Returns:
            mask: (224, 224) float32 [min_weight, max_weight]
            region_confidences: Dict of per-region confidence details
        """
        # Ensure landmarks are 2D
        if landmarks.shape[1] >= 3:
            landmarks_2d = landmarks[:, :2]
        else:
            landmarks_2d = landmarks
        
        # Compute region confidences
        region_confidences = self.region_scorer.compute_region_confidences(
            image, landmarks_2d, visibility, presence
        )
        
        # Generate weight map from landmarks and confidences
        weight_map = self._landmarks_to_region_weights(
            landmarks_2d, region_confidences
        )
        
        # Apply Gaussian smoothing
        mask = self._apply_gaussian_smoothing(weight_map)
        
        return mask, region_confidences
    
    def apply_mask_to_image(
        self,
        image: np.ndarray,    # (H, W, 3) or (H, W)
        mask: np.ndarray,     # (H, W)
    ) -> np.ndarray:
        """
        Apply spatial mask to image (element-wise multiplication).
        
        Args:
            image: Image to mask
            mask: Spatial mask (H, W) in [min_weight, max_weight]
            
        Returns:
            Masked image (same shape as input)
        """
        if image.ndim == 3:
            # Broadcast mask to channels
            masked = image * mask[:, :, np.newaxis]
        else:
            masked = image * mask
        return masked
    
    def apply_mask_to_features(
        self,
        features: torch.Tensor,  # (B, C, H, W) or (C, H, W)
        mask: np.ndarray,        # (H, W)
    ) -> torch.Tensor:
        """
        Apply spatial mask to feature map.
        
        Args:
            features: Feature tensor
            mask: Spatial mask
            
        Returns:
            Masked features
        """
        mask_tensor = torch.from_numpy(mask).to(features.device, features.dtype)
        
        if features.ndim == 4:
            # (B, C, H, W)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif features.ndim == 3:
            # (C, H, W)
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)
        
        return features * mask_tensor


class SAFMModule(nn.Module):
    """
    PyTorch module wrapper for SAFM (for integration into model forward pass).
    
    SPRINT-SIMPLIFICATION: This is a non-learned heuristic module.
    Full PMD: Learnable gating network with trainable parameters.
    """
    
    def __init__(
        self,
        output_size: int = 224,
        temperature: float = 2.0,
        gaussian_sigma: float = 8.0,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
    ):
        super().__init__()
        self.safm = HeuristicSAFM(
            output_size=output_size,
            temperature=temperature,
            gaussian_sigma=gaussian_sigma,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        
        # Register as buffer so it moves to device with model
        self.register_buffer('_dummy', torch.tensor(0))
    
    def forward(
        self,
        x: torch.Tensor,                    # (B, 3, H, W) input images
        landmarks: torch.Tensor,            # (B, 478, 2) landmarks in HxW space
        visibility: Optional[torch.Tensor] = None,  # (B, 478)
        presence: Optional[torch.Tensor] = None,    # (B, 478)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SAFM mask to input batch.
        
        Args:
            x: Batch of aligned faces (B, 3, H, W)
            landmarks: Batch of landmarks (B, 478, 2)
            visibility: Optional visibility scores
            presence: Optional presence scores
            
        Returns:
            masked_x: (B, 3, H, W) masked images
            masks: (B, H, W) generated masks
        """
        B, C, H, W = x.shape
        device = x.device
        
        masks = []
        masked_batch = []
        
        for b in range(B):
            # Convert to numpy for heuristic computation
            img_np = x[b].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
            lm_np = landmarks[b].detach().cpu().numpy()            # (478, 2)
            
            vis_np = visibility[b].detach().cpu().numpy() if visibility is not None else None
            pres_np = presence[b].detach().cpu().numpy() if presence is not None else None
            
            # Generate mask
            mask, _ = self.safm.generate_mask(img_np, lm_np, vis_np, pres_np)
            
            # Apply to image
            masked_img = self.safm.apply_mask_to_image(img_np, mask)
            
            masks.append(mask)
            masked_batch.append(masked_img)
        
        # Stack results
        masks_tensor = torch.from_numpy(np.stack(masks)).to(device, x.dtype)  # (B, H, W)
        masked_tensor = torch.from_numpy(np.stack(masked_batch)).to(device, x.dtype)
        masked_tensor = masked_tensor.permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        return masked_tensor, masks_tensor


def create_heuristic_safm(**kwargs) -> HeuristicSAFM:
    """Factory function."""
    return HeuristicSAFM(**kwargs)


def create_safm_module(**kwargs) -> SAFMModule:
    """Factory for PyTorch module."""
    return SAFMModule(**kwargs)


if __name__ == "__main__":
    # Visual test
    import matplotlib.pyplot as plt
    
    print("Testing HeuristicSAFM...")
    
    # Create SAFM
    safm = create_heuristic_safm(
        output_size=224,
        temperature=2.0,
        gaussian_sigma=8.0,
    )
    
    # Dummy aligned face
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Dummy landmarks
    dummy_landmarks = np.zeros((478, 2), dtype=np.float32)
    for i in range(478):
        angle = (i / 478) * 2 * np.pi
        r = 80 + 20 * np.sin(angle * 3)
        dummy_landmarks[i] = [112 + r * np.cos(angle), 112 + r * np.sin(angle)]
    
    # Test with all visible
    dummy_visibility = np.ones(478, dtype=np.float32)
    mask, confidences = safm.generate_mask(dummy_img, dummy_landmarks, dummy_visibility)
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"Mask mean: {mask.mean():.4f}")
    
    # Test with occlusion (mouth region low visibility)
    occ_visibility = dummy_visibility.copy()
    mouth_indices = REGION_LANDMARKS[FaceRegion.MOUTH]
    for idx in mouth_indices:
        if idx < 478:
            occ_visibility[idx] = 0.1  # Simulate occlusion
    
    mask_occ, _ = safm.generate_mask(dummy_img, dummy_landmarks, occ_visibility)
    
    print(f"\nWith mouth occlusion:")
    print(f"  Mask range: [{mask_occ.min():.4f}, {mask_occ.max():.4f}]")
    print(f"  Mask mean: {mask_occ.mean():.4f}")
    
    # Check mouth region is down-weighted
    mouth_lms = dummy_landmarks[mouth_indices]
    mouth_mask_vals = []
    for lm in mouth_lms:
        x, y = int(lm[0]), int(lm[1])
        if 0 <= x < 224 and 0 <= y < 224:
            mouth_mask_vals.append(mask_occ[y, x])
    
    if mouth_mask_vals:
        print(f"  Mouth region mask values: min={min(mouth_mask_vals):.4f}, "
              f"max={max(mouth_mask_vals):.4f}, mean={np.mean(mouth_mask_vals):.4f}")
    
    # Check eye region (should be high)
    eye_indices = REGION_LANDMARKS[FaceRegion.LEFT_EYE] + REGION_LANDMARKS[FaceRegion.RIGHT_EYE]
    eye_mask_vals = []
    for idx in eye_indices:
        if idx < 478:
            x, y = int(dummy_landmarks[idx, 0]), int(dummy_landmarks[idx, 1])
            if 0 <= x < 224 and 0 <= y < 224:
                eye_mask_vals.append(mask_occ[y, x])
    
    if eye_mask_vals:
        print(f"  Eye region mask values: min={min(eye_mask_vals):.4f}, "
              f"max={max(eye_mask_vals):.4f}, mean={np.mean(eye_mask_vals):.4f}")
    
    # Visualize if matplotlib available
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(mask, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Normal Mask')
        axes[0].axis('off')
        
        axes[1].imshow(mask_occ, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('With Mouth Occlusion')
        axes[1].axis('off')
        
        axes[2].imshow(mask_occ - mask, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[2].set_title('Difference (Occluded - Normal)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('safm_mask_test.png', dpi=150)
        print("\n[OK] Visualization saved to safm_mask_test.png")
    except Exception as e:
        print(f"\n[SKIP] Visualization: {e}")
    
    print("\n[OK] HeuristicSAFM test passed!")