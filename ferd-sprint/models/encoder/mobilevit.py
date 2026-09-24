"""
Mobile-ViT Spatial Encoder for FERD Sprint
Loads pretrained MobileViT-XS via timm, exposes 256-dim embedding layer.

SPRINT-SIMPLIFICATION: Uses pretrained ImageNet weights, frozen or lightly fine-tuned.
Full PMD: Train on AffectNet-wild + DFEW combined (teacher), then distill to MobileViT-XXS student.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MobileViTEncoder(nn.Module):
    """
    MobileViT encoder wrapper that extracts penultimate embeddings.
    
    PMD Specification:
    - Backbone: MobileViT-XS (teacher) or MobileViT-XXS (student)
    - Output embedding: 256-dimensional per frame
    - Input: 224x224 RGB face crops, normalized [0,1] or ImageNet stats
    """
    
    # Model configurations per PMD
    CONFIGS = {
        "mobilevit_xs": {
            "timm_name": "mobilevit_xs",
            "embed_dim": 384,  # MobileViT-XS final feature dim before classifier
            "target_embed_dim": 256,  # PMD target embedding dimension
        },
        "mobilevit_xxs": {
            "timm_name": "mobilevit_xxs",
            "embed_dim": 320,  # MobileViT-XXS final feature dim
            "target_embed_dim": 256,
        },
        "mobilevit_s": {
            "timm_name": "mobilevit_s",
            "embed_dim": 512,  # MobileViT-S final feature dim
            "target_embed_dim": 256,
        },
    }
    
    def __init__(
        self,
        model_variant: str = "mobilevit_xs",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        target_embed_dim: int = 256,
        dropout: float = 0.0,
    ):
        """
        Args:
            model_variant: One of "mobilevit_xs", "mobilevit_xxs", "mobilevit_s"
            pretrained: Load ImageNet pretrained weights
            freeze_backbone: Freeze encoder weights (fine-tune head only)
            target_embed_dim: Output embedding dimension (PMD: 256)
            dropout: Dropout on embedding projection
        """
        super().__init__()
        
        if model_variant not in self.CONFIGS:
            raise ValueError(f"Unknown variant: {model_variant}. Choose from {list(self.CONFIGS.keys())}")
        
        self.model_variant = model_variant
        self.config = self.CONFIGS[model_variant]
        self.target_embed_dim = target_embed_dim
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained model from timm
        logger.info(f"Loading {self.config['timm_name']} (pretrained={pretrained})...")
        self.backbone = timm.create_model(
            self.config["timm_name"],
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="",  # No pooling - we'll handle it
        )
        
        # Get the actual feature dimension
        # MobileViT in timm returns (B, C, H, W) feature map before global pool
        # We need to add adaptive pooling + projection to target_embed_dim
        self.backbone_embed_dim = self.config["embed_dim"]
        
        # Global average pooling to get 1x1 feature map
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to target embedding dimension (256 per PMD)
        self.embed_projection = nn.Sequential(
            nn.Linear(self.backbone_embed_dim, target_embed_dim),
            nn.LayerNorm(target_embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze()
            logger.info("Backbone frozen - only projection head trainable")
        else:
            logger.info("Backbone unfrozen - full fine-tuning enabled")
    
    def freeze(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_blocks(self, n: int = 2):
        """Unfreeze last n blocks of the backbone for light fine-tuning."""
        # MobileViT structure in timm: backbone.stages (list of stages)
        # Each stage contains blocks
        if hasattr(self.backbone, 'stages'):
            stages = self.backbone.stages
            # Unfreeze last n stages
            for stage in stages[-n:]:
                for param in stage.parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze last {n} stages for fine-tuning")
        else:
            # Fallback: unfreeze all
            self.unfreeze()
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature map from backbone (before global pooling).
        
        Args:
            x: (B, 3, 224, 224) input tensor
            
        Returns:
            Feature map: (B, C, H, W) where C = backbone_embed_dim
        """
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract 256-dim embedding.
        
        Args:
            x: (B, 3, 224, 224) input tensor, range [0,1] or normalized
            
        Returns:
            embeddings: (B, 256) embedding vectors
        """
        # Extract feature map: (B, C, H, W)
        features = self.forward_features(x)
        
        # Global average pooling: (B, C, 1, 1)
        pooled = self.global_pool(features)
        
        # Flatten: (B, C)
        pooled = pooled.flatten(1)
        
        # Project to target dimension: (B, 256)
        embeddings = self.embed_projection(pooled)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Return output embedding dimension."""
        return self.target_embed_dim
    
    def get_backbone_embed_dim(self) -> int:
        """Return backbone feature dimension before projection."""
        return self.backbone_embed_dim


def create_mobilevit_encoder(
    variant: str = "mobilevit_xs",
    pretrained: bool = True,
    freeze: bool = True,
    target_dim: int = 256,
) -> MobileViTEncoder:
    """Factory function for creating MobileViT encoder."""
    return MobileViTEncoder(
        model_variant=variant,
        pretrained=pretrained,
        freeze_backbone=freeze,
        target_embed_dim=target_dim,
    )


if __name__ == "__main__":
    # Test encoder
    import numpy as np
    from preprocessing.face_align import align_face
    import cv2
    
    print("Testing MobileViTEncoder...")
    
    # Create encoder
    encoder = create_mobilevit_encoder("mobilevit_xs", pretrained=True, freeze=True)
    encoder.eval()
    
    print(f"Model variant: {encoder.model_variant}")
    print(f"Backbone embed dim: {encoder.get_backbone_embed_dim()}")
    print(f"Output embed dim: {encoder.get_embedding_dim()}")
    print(f"Trainable params: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
    print(f"Total params: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        embed = encoder(dummy_input)
    
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Output embedding shape: {embed.shape}")
    print(f"Expected: (1, 256)")
    
    # Test determinism
    embed2 = encoder(dummy_input)
    diff = (embed - embed2).abs().max().item()
    print(f"Determinism check (max diff): {diff}")
    assert diff < 1e-6, "Embeddings not deterministic!"
    print("[OK] Deterministic")
    
    # Test with real aligned face
    print("\n--- Testing with real aligned face ---")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        result = align_face(frame)
        if result.success:
            # Convert to tensor: (224, 224, 3) -> (1, 3, 224, 224)
            aligned = result.aligned_face  # (224, 224, 3) float32 [0,1]
            tensor = torch.from_numpy(aligned).permute(2, 0, 1).unsqueeze(0)
            
            # Normalize with ImageNet stats (timm default)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            
            with torch.no_grad():
                embed_real = encoder(tensor)
            
            print(f"Real face embedding shape: {embed_real.shape}")
            print(f"Embedding norm: {embed_real.norm(dim=1).item():.4f}")
            print(f"Embedding range: [{embed_real.min().item():.4f}, {embed_real.max().item():.4f}]")
        else:
            print(f"Face alignment failed: {result.error}")
    else:
        print("No webcam available")
    
    print("\n[OK] Encoder test passed!")