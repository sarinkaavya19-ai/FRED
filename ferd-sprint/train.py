"""
Training Script for FERD Sprint - Block 1.7
Fine-tunes classification head + GRU head on FER2013 with frozen encoder.

SPRINT-SIMPLIFICATION: Encoder frozen, synthetic 16-frame windows, small dataset.
Full PMD: Full teacher training on AffectNet-wild + DFEW, then distillation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import time
import json
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import create_ferd_pipeline
from data.dataset import create_dataloaders, FER2013Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FERDTrainer:
    """
    Trainer for FERD fine-tuning (classification head + GRU only).
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        log_interval: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_interval = log_interval
        self.grad_clip = grad_clip
        self.use_amp = use_amp and device.type == 'cuda'
        
        # Only train classification head + GRU (encoder frozen)
        self.trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        
        self.optimizer = optim.AdamW(
            self.trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10,  # 10 epochs
            eta_min=learning_rate * 0.01,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.step = 0
        self.epoch = 0
        
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.trainable_params):,}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Device: {device}, AMP: {self.use_amp}")
    
    def train_step(self, batch: Dict) -> Tuple[float, float]:
        """Single training step."""
        self.model.train()
        
        frames = batch['frames'].to(self.device)      # (B, 16, 3, 224, 224)
        labels = batch['label'].to(self.device)       # (B,)
        landmarks = batch['landmarks'].to(self.device) # (B, 16, 478, 2)
        visibility = batch['visibility'].to(self.device) # (B, 16, 478)
        
        B, T, C, H, W = frames.shape
        
        # Reshape for pipeline: process each frame independently through encoder+SAFM
        # Pipeline expects (B, 3, 224, 224) per step
        # For training, we can process the full sequence
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast():
                loss, acc = self._compute_loss(frames, labels, landmarks, visibility)
            self.scaler.scale(loss).backward()
            
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.trainable_params, self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, acc = self._compute_loss(frames, labels, landmarks, visibility)
            loss.backward()
            
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.trainable_params, self.grad_clip)
            
            self.optimizer.step()
        
        self.scheduler.step()
        self.step += 1
        
        return loss.item(), acc
    
    def _compute_loss(
        self, 
        frames: torch.Tensor,      # (B, T, 3, H, W)
        labels: torch.Tensor,      # (B,)
        landmarks: torch.Tensor,   # (B, T, 478, 2)
        visibility: torch.Tensor,  # (B, T, 478)
    ) -> Tuple[torch.Tensor, float]:
        """Compute loss for batch."""
        B, T, C, H, W = frames.shape
        
        # Process through pipeline
        # We need to handle the temporal dimension
        # For each sample in batch, process its 16 frames through the pipeline
        
        all_logits = []
        
        for b in range(B):
            # Process sequence for this sample
            sample_frames = frames[b]      # (T, 3, H, W)
            sample_landmarks = landmarks[b] # (T, 478, 2)
            sample_vis = visibility[b]      # (T, 478)
            
            # Get embeddings for all frames
            embeddings = []
            for t in range(T):
                with torch.set_grad_enabled(self.model.encoder.training):
                    # SAFM + Encoder
                    if self.model.use_safm:
                        masked_frame, _ = self.model.safm(
                            sample_frames[t:t+1],  # (1, 3, H, W)
                            sample_landmarks[t:t+1], # (1, 478, 2)
                            sample_vis[t:t+1],       # (1, 478)
                        )
                    else:
                        masked_frame = sample_frames[t:t+1]
                    
                    embed = self.model.encoder(masked_frame)  # (1, 256)
                    embeddings.append(embed)
            
            embeddings = torch.cat(embeddings, dim=0)  # (T, 256)
            
            # GRU
            gru_out, _ = self.model.gru(embeddings.unsqueeze(0))  # (1, T, 256)
            
            # Classification (use last timestep)
            logits = self.model.classifier(gru_out)  # (1, 7) - takes last timestep internally
            all_logits.append(logits)  # (1, 7)
        
        logits = torch.cat(all_logits, dim=0)  # (B, 7)
        
        # Loss
        loss = self.criterion(logits, labels)
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        
        return loss, acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            visibility = batch['visibility'].to(self.device)
            
            B, T, C, H, W = frames.shape
            
            all_logits = []
            
            for b in range(B):
                sample_frames = frames[b]
                sample_landmarks = landmarks[b]
                sample_vis = visibility[b]
                
                embeddings = []
                for t in range(T):
                    if self.model.use_safm:
                        masked_frame, _ = self.model.safm(
                            sample_frames[t:t+1],
                            sample_landmarks[t:t+1],
                            sample_vis[t:t+1],
                        )
                    else:
                        masked_frame = sample_frames[t:t+1]
                    
                    embed = self.model.encoder(masked_frame)
                    embeddings.append(embed)
                
                embeddings = torch.cat(embeddings, dim=0)
                gru_out, _ = self.model.gru(embeddings.unsqueeze(0))
                logits = self.model.classifier(gru_out)  # (1, 7)
                all_logits.append(logits)
            
            logits = torch.cat(all_logits, dim=0)
            loss = self.criterion(logits, labels)
            
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            loss, acc = self.train_step(batch)
            
            total_loss += loss
            total_acc += acc
            num_batches += 1
            
            self.train_losses.append(loss)
            self.train_accs.append(acc)
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{acc:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                })
            
            # Check for NaN
            if np.isnan(loss):
                logger.error(f"NaN loss detected at step {self.step}!")
                return float('nan'), 0.0
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def fit(self, epochs: int = 10, patience: int = 3) -> Dict:
        """Full training loop with early stopping."""
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        no_improve = 0
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            if np.isnan(train_loss):
                logger.error("Training stopped due to NaN loss")
                break
            
            # Validate
            val_loss, val_acc = self.validate()
            
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"time={epoch_time:.1f}s, total={total_time:.1f}s"
            )
            
            # Check improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                no_improve = 0
                # Save best model
                self.save_checkpoint('best_model.pt')
                logger.info(f"  -> New best val_acc: {val_acc:.4f}")
            else:
                no_improve += 1
                logger.info(f"  -> No improvement for {no_improve} epochs")
            
            # Early stopping
            if no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'total_time': total_time,
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'step': self.step,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, path)
        logger.info(f"Checkpoint saved to {path}")


def create_model_for_training(
    encoder_variant: str = "mobilevit_xs",
    encoder_pretrained: bool = True,
    encoder_frozen: bool = True,
    unfreeze_last_n: int = 0,
    gru_hidden_dim: int = 256,
    gru_num_layers: int = 2,
    class_hidden_dim: int = 128,
    dropout: float = 0.1,
    use_safm: bool = True,
) -> nn.Module:
    """Create model with proper trainable/frozen setup."""
    model = create_ferd_pipeline(
        encoder_variant=encoder_variant,
        encoder_pretrained=encoder_pretrained,
        encoder_frozen=encoder_frozen,
        gru_hidden_dim=gru_hidden_dim,
        gru_num_layers=gru_num_layers,
        class_hidden_dim=class_hidden_dim,
        class_dropout=dropout,
        use_safm=use_safm,
    )
    
    # Unfreeze last N encoder blocks if requested
    if unfreeze_last_n > 0 and hasattr(model.encoder, 'unfreeze_last_n_blocks'):
        model.encoder.unfreeze_last_n_blocks(unfreeze_last_n)
        logger.info(f"Unfroze last {unfreeze_last_n} encoder blocks")
    
    return model


def main():
    """Main training entry point."""
    # Config
    config = {
        'split_file': 'data/datasets/fer2013/processed/split.json',
        'batch_size': 8,
        'window_size': 16,
        'max_samples_per_class': 500,  # Cap for speed
        'encoder_variant': 'mobilevit_xs',
        'encoder_pretrained': True,
        'encoder_frozen': True,
        'unfreeze_last_n': 0,  # Set to 1-2 for light fine-tuning
        'gru_hidden_dim': 256,
        'gru_num_layers': 2,
        'class_hidden_dim': 128,
        'dropout': 0.1,
        'use_safm': True,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'epochs': 20,
        'patience': 5,
        'log_interval': 10,
        'use_amp': True,
        'num_workers': 0,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        split_file=config['split_file'],
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        max_samples_per_class=config['max_samples_per_class'],
        num_workers=config['num_workers'],
        use_weighted_sampler=True,
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model_for_training(
        encoder_variant=config['encoder_variant'],
        encoder_pretrained=config['encoder_pretrained'],
        encoder_frozen=config['encoder_frozen'],
        unfreeze_last_n=config['unfreeze_last_n'],
        gru_hidden_dim=config['gru_hidden_dim'],
        gru_num_layers=config['gru_num_layers'],
        class_hidden_dim=config['class_hidden_dim'],
        dropout=config['dropout'],
        use_safm=config['use_safm'],
    )
    
    # Create trainer
    trainer = FERDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        grad_clip=config['grad_clip'],
        use_amp=config['use_amp'],
        log_interval=config['log_interval'],
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.fit(
        epochs=config['epochs'],
        patience=config['patience'],
    )
    
    # Save final checkpoint
    trainer.save_checkpoint('final_model.pt')
    
    # Save history
    with open('training_history.json', 'w') as f:
        # Convert numpy/torch to python types
        hist = {}
        for k, v in history.items():
            if isinstance(v, (list, tuple)):
                hist[k] = [float(x) if isinstance(x, (np.floating, torch.Tensor)) else x for x in v]
            elif isinstance(v, (np.floating, torch.Tensor)):
                hist[k] = float(v)
            else:
                hist[k] = v
        json.dump(hist, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Best val accuracy: {history['best_val_acc']:.4f}")
    
    return history


if __name__ == "__main__":
    main()