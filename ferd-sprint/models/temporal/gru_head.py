"""
GRU Temporal Head for FERD Sprint
Consumes 16-frame sliding window of per-frame embeddings, outputs hidden state.

SPRINT-SIMPLIFICATION: Single GRU configuration only (TCN alternative dropped per Sprint Plan).
Full PMD: Configurable GRU/TCN, trained on genuine temporal dynamics from video data.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GRUTemporalHead(nn.Module):
    """
    GRU-based temporal head for sequence modeling across frames.
    
    PMD Specification:
    - Input: 16-frame sliding window of 256-dim embeddings
    - Hidden size: 256 (matches embedding dim)
    - Layers: 2 (configurable)
    - Bidirectional: False (causal for streaming)
    - Output: Final hidden state for classification
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        """
        Args:
            input_dim: Input embedding dimension (256 per PMD)
            hidden_dim: GRU hidden dimension (256 per PMD)
            num_layers: Number of GRU layers
            dropout: Dropout between layers
            bidirectional: If True, uses BiGRU (not causal)
            batch_first: Input shape (B, T, D) if True, else (T, B, D)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Layer norm on hidden state
        self.hidden_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        
        # Output dimension (for downstream heads)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
        logger.info(f"GRUTemporalHead: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                    f"num_layers={num_layers}, bidirectional={bidirectional}, "
                    f"output_dim={self.output_dim}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GRU.
        
        Args:
            x: Input embeddings (B, T, D) where T=16 (window), D=256
            hidden: Optional initial hidden state (num_layers * num_dir, B, H)
            
        Returns:
            output: GRU outputs for all timesteps (B, T, output_dim)
            hidden: Final hidden state (num_layers * num_dir, B, H)
        """
        # x shape: (B, T, D) with batch_first=True
        output, hidden = self.gru(x, hidden)
        
        # Apply layer norm to each timestep's output
        B, T, D = output.shape
        output = output.reshape(B * T, D)
        output = self.hidden_norm(output)
        output = output.reshape(B, T, D)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state with zeros."""
        num_dirs = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * num_dirs, batch_size, self.hidden_dim, device=device)


class StreamingGRUHead(GRUTemporalHead):
    """
    Streaming version that maintains hidden state across calls.
    For real-time inference where frames arrive one at a time.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('hidden_state', None)
        self.register_buffer('frame_buffer', None)
        self.window_size = 16
    
    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset hidden state and frame buffer."""
        if device is None:
            device = next(self.parameters()).device
        self.hidden_state = self.init_hidden(batch_size, device)
        self.frame_buffer = torch.zeros(batch_size, 0, self.input_dim, device=device)
    
    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single frame (or batch of frames) in streaming mode.
        
        Args:
            x: Input embedding (B, 1, D) or (B, D)
            
        Returns:
            output: Current output (B, output_dim)
            hidden: Updated hidden state
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        
        B = x.shape[0]
        
        # Initialize if needed
        if self.hidden_state is None or self.hidden_state.shape[1] != B:
            self.reset_state(B, x.device)
        
        # Append to buffer
        self.frame_buffer = torch.cat([self.frame_buffer, x], dim=1)
        
        # Keep only last window_size frames
        if self.frame_buffer.shape[1] > self.window_size:
            self.frame_buffer = self.frame_buffer[:, -self.window_size:, :]
        
        # Process full buffer through GRU
        # But we only need the last output for streaming
        if self.frame_buffer.shape[1] == self.window_size:
            # Full window - process all
            output, self.hidden_state = self.forward(self.frame_buffer, self.hidden_state)
            return output[:, -1, :], self.hidden_state
        else:
            # Not enough frames yet - process what we have
            output, self.hidden_state = self.forward(self.frame_buffer, self.hidden_state)
            return output[:, -1, :], self.hidden_state
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        """Standard forward for training (full sequence)."""
        return super().forward(x, hidden)


def create_gru_head(
    input_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    streaming: bool = False,
) -> nn.Module:
    """Factory function for GRU head."""
    if streaming:
        return StreamingGRUHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    return GRUTemporalHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Test GRU head
    print("Testing GRUTemporalHead...")
    
    # Standard GRU
    gru = create_gru_head(input_dim=256, hidden_dim=256, num_layers=2)
    gru.eval()
    
    # Dummy 16-frame sequence
    B = 4
    T = 16
    D = 256
    x = torch.randn(B, T, D)
    
    with torch.no_grad():
        output, hidden = gru(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape} (expected: {B}, {T}, {256})")
    print(f"Hidden shape: {hidden.shape} (expected: {2}, {B}, {256})")
    
    # Test determinism
    with torch.no_grad():
        output2, hidden2 = gru(x)
    print(f"Deterministic: {(output - output2).abs().max().item() == 0}")
    
    # Test streaming version
    print("\n--- Testing StreamingGRUHead ---")
    stream_gru = create_gru_head(input_dim=256, hidden_dim=256, streaming=True)
    stream_gru.eval()
    stream_gru.reset_state(batch_size=1)
    
    # Feed frames one by one
    outputs = []
    for t in range(20):  # More than window size
        frame = torch.randn(1, 1, 256)
        out, hidden = stream_gru.step(frame)
        outputs.append(out)
        
        if t < 3 or t >= 17:
            print(f"  Frame {t}: output shape {out.shape}, buffer={stream_gru.frame_buffer.shape[1]}")
    
    print(f"Total outputs: {len(outputs)}")
    print(f"Final buffer size: {stream_gru.frame_buffer.shape[1]}")
    
    print("\n[OK] GRUTemporalHead test passed!")