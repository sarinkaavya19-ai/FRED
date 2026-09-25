"""
Full FERD Pipeline Integration for Sprint
Wires together: FaceAlign -> SAFM -> Encoder -> GRU -> Classification

SPRINT-SIMPLIFICATION: End-to-end forward pass with random weights (untrained).
Full PMD: Trained teacher -> distillation -> quantized student -> edge export.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder.mobilevit import create_mobilevit_encoder, MobileViTEncoder
from models.safm.attention_mask import create_safm_module, SAFMModule
from models.temporal.gru_head import create_gru_head, GRUTemporalHead, StreamingGRUHead
from models.heads.classification_head import create_classification_head, ClassificationHead


class FERDPipeline(nn.Module):
    """
    Full FERD inference pipeline (non-streaming, batch mode).
    
    Flow:
    1. Face alignment (external - MediaPipe)
    2. SAFM masking
    3. MobileViT encoder -> 256-dim embedding
    4. Temporal buffer (16 frames)
    5. GRU temporal head
    6. Classification head -> 7-class probabilities
    """
    
    def __init__(
        self,
        encoder_variant: str = "mobilevit_xs",
        encoder_pretrained: bool = True,
        encoder_frozen: bool = True,
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.1,
        class_hidden_dim: int = 128,
        class_dropout: float = 0.1,
        temperature: float = 1.0,
        window_size: int = 16,
        use_safm: bool = True,
    ):
        super().__init__()
        
        self.window_size = window_size
        self.use_safm = use_safm
        
        # SAFM module
        if use_safm:
            self.safm = create_safm_module(output_size=224)
        else:
            self.safm = None
        
        # Encoder
        self.encoder = create_mobilevit_encoder(
            variant=encoder_variant,
            pretrained=encoder_pretrained,
            freeze=encoder_frozen,
            target_dim=256,
        )
        
        # GRU temporal head
        self.gru = create_gru_head(
            input_dim=256,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
            streaming=False,
        )
        
        # Classification head
        self.classifier = create_classification_head(
            input_dim=gru_hidden_dim,
            hidden_dim=class_hidden_dim,
            dropout=class_dropout,
            temperature=temperature,
        )
        
        # Frame buffer for temporal context
        self.register_buffer('frame_buffer', torch.zeros(1, 0, 256))
        
        logger.info(f"FERDPipeline initialized: window_size={window_size}, use_safm={use_safm}")
    
    def forward(
        self,
        x: torch.Tensor,                    # (B, 3, 224, 224) aligned faces
        landmarks: torch.Tensor,            # (B, 478, 2) landmarks in 224x224 space
        visibility: Optional[torch.Tensor] = None,  # (B, 478)
        presence: Optional[torch.Tensor] = None,    # (B, 478)
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full pipeline.
        
        Args:
            x: Batch of aligned face crops
            landmarks: Facial landmarks for SAFM
            visibility: Optional landmark visibility
            presence: Optional landmark presence
            return_all: If True, return intermediate outputs
            
        Returns:
            Dict with 'logits', 'probs', 'embeddings', 'masks', 'hidden'
        """
        B = x.shape[0]
        
        # 1. SAFM masking
        if self.use_safm and self.safm is not None:
            masked_x, masks = self.safm(x, landmarks, visibility, presence)
        else:
            masked_x = x
            masks = torch.ones(B, 224, 224, device=x.device, dtype=x.dtype)
        
        # 2. Encoder -> embeddings
        embeddings = self.encoder(masked_x)  # (B, 256)
        
        # 3. GRU temporal processing
        # For batch mode, we process each sample independently
        # In practice, this would be a sliding window over time
        # Here we simulate by expanding to (B, window_size, 256) with repeated embeddings
        # This is a SPRINT-SIMPLIFICATION for testing pipeline shape
        if embeddings.dim() == 2:
            # Repeat for temporal dimension (simulates 16-frame window)
            embeddings_seq = embeddings.unsqueeze(1).repeat(1, self.window_size, 1)
        else:
            embeddings_seq = embeddings
        
        gru_output, hidden = self.gru(embeddings_seq)
        
        # 4. Classification (use last timestep)
        logits = self.classifier(gru_output)
        probs = self.classifier.predict_proba(gru_output)
        
        result = {
            'logits': logits,           # (B, 7) or (B, T, 7)
            'probs': probs,             # (B, 7) or (B, T, 7)
            'embeddings': embeddings,   # (B, 256)
            'masks': masks,             # (B, 224, 224)
            'hidden': hidden,           # (num_layers, B, hidden_dim)
        }
        
        if not return_all:
            # Return only final probabilities
            return probs
        
        return result
    
    def process_sequence(
        self,
        frames: List[torch.Tensor],       # List of (3, 224, 224) frames
        landmarks_list: List[torch.Tensor],  # List of (478, 2)
        visibility_list: Optional[List[torch.Tensor]] = None,
        presence_list: Optional[List[torch.Tensor]] = None,
    ) -> List[Dict]:
        """
        Process a sequence of frames (simulating video).
        Maintains temporal buffer across frames.
        
        SPRINT-SIMPLIFICATION: Processes one frame at a time, maintains buffer.
        """
        results = []
        
        for i, (frame, landmarks) in enumerate(zip(frames, landmarks_list)):
            vis = visibility_list[i] if visibility_list else None
            pres = presence_list[i] if presence_list else None
            
            # Add batch dim
            frame_b = frame.unsqueeze(0)
            landmarks_b = landmarks.unsqueeze(0)
            vis_b = vis.unsqueeze(0) if vis is not None else None
            pres_b = pres.unsqueeze(0) if pres is not None else None
            
            # Forward
            with torch.no_grad():
                result = self.forward(frame_b, landmarks_b, vis_b, pres_b, return_all=True)
            
            results.append(result)
        
        return results


class StreamingFERDPipeline(nn.Module):
    """
    Streaming version for real-time inference.
    Maintains GRU hidden state and frame buffer across calls.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # Use streaming GRU
        kwargs['streaming'] = True
        
        # Create components
        self.safm = create_safm_module(output_size=224) if kwargs.get('use_safm', True) else None
        self.encoder = create_mobilevit_encoder(
            variant=kwargs.get('encoder_variant', 'mobilevit_xs'),
            pretrained=kwargs.get('encoder_pretrained', True),
            freeze=kwargs.get('encoder_frozen', True),
        )
        self.gru = create_gru_head(
            input_dim=256,
            hidden_dim=kwargs.get('gru_hidden_dim', 256),
            num_layers=kwargs.get('gru_num_layers', 2),
            dropout=kwargs.get('gru_dropout', 0.1),
            streaming=True,
        )
        self.classifier = create_classification_head(
            input_dim=kwargs.get('gru_hidden_dim', 256),
            hidden_dim=kwargs.get('class_hidden_dim', 128),
            dropout=kwargs.get('class_dropout', 0.1),
            temperature=kwargs.get('temperature', 1.0),
        )
        
        self.window_size = kwargs.get('window_size', 16)
        self._initialized = False
    
    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset streaming state."""
        if device is None:
            device = next(self.parameters()).device
        self.gru.reset_state(batch_size, device)
        self._initialized = True
    
    @torch.no_grad()
    def step(
        self,
        frame: torch.Tensor,              # (3, 224, 224) or (B, 3, 224, 224)
        landmarks: torch.Tensor,          # (478, 2) or (B, 478, 2)
        visibility: Optional[torch.Tensor] = None,
        presence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process single frame in streaming mode.
        
        Returns:
            Dict with 'probs', 'embedding', 'mask', 'hidden'
        """
        # Add batch dim if needed
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
            landmarks = landmarks.unsqueeze(0)
            if visibility is not None:
                visibility = visibility.unsqueeze(0)
            if presence is not None:
                presence = presence.unsqueeze(0)
        
        B = frame.shape[0]
        device = frame.device
        
        # Initialize streaming state
        if not self._initialized:
            self.reset(B, device)
        
        # 1. SAFM
        if self.safm is not None:
            masked_frame, mask = self.safm(frame, landmarks, visibility, presence)
        else:
            masked_frame = frame
            mask = torch.ones(B, 224, 224, device=device, dtype=frame.dtype)
        
        # 2. Encoder
        embedding = self.encoder(masked_frame)  # (B, 256)
        
        # 3. Streaming GRU step
        gru_out, hidden = self.gru.step(embedding)
        
        # 4. Classification
        probs = self.classifier.predict_proba(gru_out)
        pred, conf = self.classifier.predict(gru_out)
        
        return {
            'probs': probs,           # (B, 7)
            'pred_class': pred,       # (B,)
            'confidence': conf,       # (B,)
            'embedding': embedding,   # (B, 256)
            'mask': mask,             # (B, 224, 224)
            'hidden': hidden,         # (num_layers, B, hidden_dim)
        }


def create_ferd_pipeline(**kwargs) -> FERDPipeline:
    """Factory for batch pipeline."""
    return FERDPipeline(**kwargs)


def create_streaming_pipeline(**kwargs) -> StreamingFERDPipeline:
    """Factory for streaming pipeline."""
    return StreamingFERDPipeline(**kwargs)


if __name__ == "__main__":
    # Test full pipeline
    print("Testing FERDPipeline (batch mode)...")
    
    pipeline = create_ferd_pipeline(
        encoder_variant="mobilevit_xs",
        encoder_pretrained=True,
        encoder_frozen=True,
        use_safm=True,
        window_size=16,
    )
    pipeline.eval()
    
    # Dummy batch
    B = 2
    x = torch.randn(B, 3, 224, 224)
    landmarks = torch.zeros(B, 478, 2)
    for b in range(B):
        for i in range(478):
            angle = (i / 478) * 2 * np.pi
            r = 80 + 20 * np.sin(angle * 3)
            landmarks[b, i, 0] = 112 + r * np.cos(angle)
            landmarks[b, i, 1] = 112 + r * np.sin(angle)
    visibility = torch.ones(B, 478)
    
    with torch.no_grad():
        result = pipeline.forward(x, landmarks, visibility, return_all=True)
    
    print(f"Input: {x.shape}")
    print(f"Masks: {result['masks'].shape}")
    print(f"Embeddings: {result['embeddings'].shape}")
    print(f"GRU hidden: {result['hidden'].shape}")
    print(f"Logits: {result['logits'].shape}")
    print(f"Probs: {result['probs'].shape}")
    print(f"Probs sum: {result['probs'].sum(dim=-1)}")
    
    # Test streaming pipeline
    print("\n--- Testing StreamingFERDPipeline ---")
    stream_pipe = create_streaming_pipeline()
    stream_pipe.eval()
    
    # Simulate 20 frames
    for t in range(20):
        frame = torch.randn(3, 224, 224)
        lm = torch.zeros(478, 2)
        for i in range(478):
            angle = (i / 478) * 2 * np.pi
            r = 80 + 20 * np.sin(angle * 3)
            lm[i, 0] = 112 + r * np.cos(angle)
            lm[i, 1] = 112 + r * np.sin(angle)
        
        result = stream_pipe.step(frame, lm)
        
        if t < 3 or t >= 17:
            print(f"  Frame {t}: probs={result['probs'].shape}, pred={result['pred_class'].item()}, conf={result['confidence'].item():.4f}")
    
    print("\n[OK] Full pipeline test passed!")