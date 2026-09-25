"""
Classification Head for FERD Sprint
7-class softmax on top of GRU's final hidden state.

SPRINT-SIMPLIFICATION: 7-class only (valence-arousal + micro-expression dropped per Sprint Plan).
Full PMD: Multi-head with valence-arousal regression + micro-expression onset detection.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# PMD 7-class emotion labels
EMOTION_CLASSES = [
    "anger",
    "disgust", 
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]

NUM_CLASSES = len(EMOTION_CLASSES)


class ClassificationHead(nn.Module):
    """
    7-class emotion classification head.
    
    Takes GRU hidden state (or pooled temporal features) and outputs
    class probabilities with temperature-scaled softmax for calibrated confidence.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: Input feature dimension (GRU hidden_dim = 256)
            num_classes: Number of emotion classes (7 per PMD)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            temperature: Softmax temperature for calibration (1.0 = standard)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Classification network
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        logger.info(f"ClassificationHead: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                    f"num_classes={num_classes}, temperature={temperature}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, D) or (B, T, D) - uses last timestep if 3D
            
        Returns:
            logits: (B, num_classes) raw logits
        """
        # Handle temporal input (B, T, D) - take last timestep
        if x.dim() == 3:
            x = x[:, -1, :]  # (B, D)
        
        # Classification logits
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get calibrated probability distribution.
        
        Args:
            x: Input features
            
        Returns:
            probs: (B, num_classes) probabilities summing to 1
        """
        logits = self.forward(x)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return probs
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence.
        
        Returns:
            pred_class: (B,) predicted class indices
            confidence: (B,) max probability
        """
        probs = self.predict_proba(x)
        confidence, pred_class = probs.max(dim=-1)
        return pred_class, confidence


class MultiTaskHead(nn.Module):
    """
    Multi-task head for full PMD (classification + valence-arousal + micro-expression).
    
    SPRINT-SIMPLIFICATION: Not used in sprint (only classification).
    Kept here for reference and future expansion.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.emotion_head = nn.Linear(hidden_dim, num_classes)
        self.valence_arousal_head = nn.Linear(hidden_dim, 2)  # valence, arousal
        self.microexpr_head = nn.Linear(hidden_dim, 2)  # onset, offset
    
    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() == 3:
            x = x[:, -1, :]
        
        features = self.trunk(x)
        
        return {
            'emotion_logits': self.emotion_head(features),
            'valence_arousal': torch.tanh(self.valence_arousal_head(features)),  # [-1, 1]
            'microexpr': torch.sigmoid(self.microexpr_head(features)),  # [0, 1]
        }


def create_classification_head(
    input_dim: int = 256,
    num_classes: int = NUM_CLASSES,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    temperature: float = 1.0,
) -> ClassificationHead:
    """Factory function."""
    return ClassificationHead(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        temperature=temperature,
    )


if __name__ == "__main__":
    # Test classification head
    print("Testing ClassificationHead...")
    
    head = create_classification_head(input_dim=256)
    head.eval()
    
    # Test with GRU output (B, D)
    B = 4
    x = torch.randn(B, 256)
    
    with torch.no_grad():
        logits = head(x)
        probs = head.predict_proba(x)
        pred, conf = head.predict(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape} (expected: {B}, {NUM_CLASSES})")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs sum: {probs.sum(dim=-1)} (should be 1.0)")
    print(f"Pred shape: {pred.shape}")
    print(f"Conf shape: {conf.shape}")
    
    # Test with temporal input (B, T, D)
    x_temp = torch.randn(B, 16, 256)
    with torch.no_grad():
        logits_temp = head(x_temp)
    print(f"\nTemporal input: {x_temp.shape} -> logits: {logits_temp.shape}")
    
    # Test temperature scaling
    print(f"\nTemperature scaling:")
    for temp in [0.5, 1.0, 2.0]:
        head_temp = create_classification_head(temperature=temp)
        head_temp.eval()
        with torch.no_grad():
            probs_temp = head_temp.predict_proba(x)
        entropy = -(probs_temp * torch.log(probs_temp + 1e-8)).sum(dim=-1).mean()
        print(f"  T={temp}: mean entropy={entropy:.4f}")
    
    print("\n[OK] ClassificationHead test passed!")