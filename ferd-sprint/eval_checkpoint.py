"""
Quick Evaluation Script for Block 2.1
Evaluates model on validation set, computes accuracy and weighted F1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import create_streaming_pipeline
from data.dataset import create_dataloaders, FER2013Dataset

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

def evaluate_model(model, val_loader, device, max_batches=None):
    """Evaluate model on validation set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            frames = batch['frames'].to(device)       # (B, 16, 3, 224, 224)
            labels = batch['label'].to(device)        # (B,)
            landmarks = batch['landmarks'].to(device)  # (B, 16, 478, 2)
            visibility = batch['visibility'].to(device) # (B, 16, 478)
            
            B, T, C, H, W = frames.shape
            
            for b in range(B):
                sample_frames = frames[b]
                sample_landmarks = landmarks[b]
                sample_vis = visibility[b]
                
                # Get embeddings
                embeddings = []
                for t in range(T):
                    if model.safm is not None:
                        masked_frame, _ = model.safm(
                            sample_frames[t:t+1],
                            sample_landmarks[t:t+1],
                            sample_vis[t:t+1],
                        )
                    else:
                        masked_frame = sample_frames[t:t+1]
                    
                    embed = model.encoder(masked_frame)
                    embeddings.append(embed)
                
                embeddings = torch.cat(embeddings, dim=0)
                gru_out, _ = model.gru(embeddings.unsqueeze(0))
                logits = model.classifier(gru_out)  # (1, 7)
                probs = torch.softmax(logits, dim=-1)
                
                pred = probs.argmax(dim=-1).item()
                
                all_preds.append(pred)
                all_labels.append(labels[b].item())
                all_probs.append(probs.squeeze(0).cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def compute_metrics(preds, labels):
    """Compute accuracy and weighted F1."""
    acc = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Per-class accuracy
    cm = confusion_matrix(labels, preds, labels=list(range(7)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm,
    }

def spot_check_predictions(model, val_dataset, device, num_samples=10):
    """Spot-check qualitative predictions."""
    print(f"\n--- Spot-checking {num_samples} validation samples ---")
    
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            sample = val_dataset[idx]
            
            frames = sample['frames'].unsqueeze(0).to(device)      # (1, 16, 3, 224, 224)
            landmarks = sample['landmarks'].unsqueeze(0).to(device) # (1, 16, 478, 2)
            visibility = sample['visibility'].unsqueeze(0).to(device) # (1, 16, 478)
            label = sample['label'].item()
            
            # Process through model
            embeddings = []
            for t in range(16):
                if model.safm is not None:
                    masked_frame, _ = model.safm(
                        frames[0, t:t+1],
                        landmarks[0, t:t+1],
                        visibility[0, t:t+1],
                    )
                else:
                    masked_frame = frames[0, t:t+1]
                
                embed = model.encoder(masked_frame)
                embeddings.append(embed)
            
            embeddings = torch.cat(embeddings, dim=0)
            gru_out, _ = model.gru(embeddings.unsqueeze(0))
            logits = model.classifier(gru_out)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            pred = probs.argmax()
            confidence = probs.max()
            
            print(f"  Sample {idx}: True={EMOTIONS[label]:12s} | Pred={EMOTIONS[pred]:12s} ({confidence:.2%}) | Probs={np.round(probs, 3)}")

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint if exists
    checkpoint_path = 'best_model.pt'
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        print("No checkpoint found, using fresh model...")
        checkpoint = None
    
    # Create model
    model = create_streaming_pipeline(
        encoder_variant='mobilevit_xs',
        encoder_pretrained=True,
        encoder_frozen=True,
        use_safm=True,
    ).to(device)
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model.eval()
    
    # Create validation loader (small subset for speed)
    print("Creating validation dataloader...")
    _, val_loader, _ = create_dataloaders(
        split_file='data/datasets/fer2013/processed/split.json',
        batch_size=4,
        window_size=16,
        max_samples_per_class=20,  # Very small subset for quick eval
        num_workers=0,
        use_weighted_sampler=False,
    )
    
    print(f"Val batches: {len(val_loader)}")
    
    # Evaluate
    print("\n--- Running evaluation ---")
    preds, labels, probs = evaluate_model(model, val_loader, device, max_batches=5)
    
    metrics = compute_metrics(preds, labels)
    
    print(f"\n=== VALIDATION METRICS ===")
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro):   {metrics['f1_macro']:.4f}")
    print(f"Chance level:  {1/7:.4f} ({100/7:.1f}%)")
    print(f"Above chance:  {metrics['accuracy'] - 1/7:.4f} ({100*(metrics['accuracy'] - 1/7):.1f}%)")
    
    print(f"\nPer-class accuracy:")
    for i, (emotion, acc) in enumerate(zip(EMOTIONS, metrics['per_class_acc'])):
        print(f"  {emotion:12s}: {acc:.4f} ({acc*100:.1f}%)")
    
    # Check for collapse
    unique_preds = len(np.unique(preds))
    print(f"\nUnique predictions: {unique_preds}/7")
    if unique_preds == 1:
        print("⚠️  WARNING: Model collapsed to single class!")
    elif unique_preds < 4:
        print("⚠️  WARNING: Low prediction diversity!")
    else:
        print("✅ Predictions vary across classes")
    
    # Spot check
    val_dataset = FER2013Dataset(
        split_file='data/datasets/fer2013/processed/split.json',
        split='val',
        window_size=16,
        max_samples_per_class=20,
        return_landmarks=True,
    )
    spot_check_predictions(model, val_dataset, device, num_samples=5)
    
    # GO/NO-GO decision
    print(f"\n=== GO/NO-GO 4 DECISION ===")
    if metrics['accuracy'] > 0.20 and unique_preds >= 4:
        print("✅ GO: Accuracy meaningfully above chance, predictions vary")
        print("   → Proceed to Block 2.2 with this checkpoint")
    else:
        print("❌ NO-GO: Accuracy near chance or collapsed")
        print("   → Fallback: (a) unfreeze encoder layers + 2hr fine-tune, or")
        print("   → Fallback: (b) demo with partial checkpoint, narrate transparently")
    
    return metrics

if __name__ == "__main__":
    main()