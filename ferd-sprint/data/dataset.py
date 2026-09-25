"""
Training Data Preparation for FERD Sprint
PyTorch Dataset/DataLoader for FER2013 with synthetic 16-frame windows.

SPRINT-SIMPLIFICATION: Static images → synthetic 16-frame windows via repetition + augmentation.
Full PMD: Genuine video-sequence emotion data (DFEW, CASME II) for temporal training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import random
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# PMD 7-class emotion labels (matching FER2013)
EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(EMOTION_CLASSES)}

# FER2013 original labels (0-6) to our class order
# FER2013: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# Our order: angry, disgust, fear, happy, sad, surprise, neutral
# They match! But let's be explicit
FER2013_TO_PMD = [0, 1, 2, 3, 4, 5, 6]  # Direct mapping


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset with synthetic 16-frame windows for temporal training.
    
    Since FER2013 is static images, we create temporal windows by:
    1. Repeating the same frame 16 times (base)
    2. Adding light augmentation per frame (jitter, noise, flip)
    
    This is a SPRINT-SIMPLIFICATION documented in Section 7.
    """
    
    def __init__(
        self,
        split_file: str,
        split: str = "train",
        window_size: int = 16,
        image_size: int = 224,
        augment: bool = True,
        max_samples_per_class: Optional[int] = None,
        return_landmarks: bool = False,  # Not available for FER2013
    ):
        """
        Args:
            split_file: Path to split.json (from prepare_fer2013.py)
            split: "train", "val", or "test"
            window_size: Number of frames in temporal window (16 per PMD)
            image_size: Target image size (224 for MobileViT)
            augment: Whether to apply augmentation
            max_samples_per_class: Cap samples per class for balanced training
            return_landmarks: If True, return dummy landmarks (for pipeline compat)
        """
        self.split = split
        self.window_size = window_size
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.return_landmarks = return_landmarks
        
        # Load split file
        with open(split_file, 'r') as f:
            all_samples = json.load(f)
        
        # Filter by split
        self.samples = [s for s in all_samples if s['split'] == split]
        
        # Limit per class if requested
        if max_samples_per_class:
            self.samples = self._limit_per_class(self.samples, max_samples_per_class)
        
        # Build transforms
        self.transform = self._build_transforms()
        
        # Base directory for images
        self.base_dir = Path(split_file).parent
        
        logger.info(f"FER2013Dataset [{split}]: {len(self.samples)} samples, "
                    f"window_size={window_size}, augment={self.augment}")
        
        # Log class distribution
        self._log_class_distribution()
    
    def _limit_per_class(self, samples: List[Dict], max_per_class: int) -> List[Dict]:
        """Limit samples per class for balanced training."""
        class_counts = Counter()
        limited = []
        for s in samples:
            cls = s['label']
            if class_counts[cls] < max_per_class:
                limited.append(s)
                class_counts[cls] += 1
        logger.info(f"Limited to {max_per_class} per class: {dict(class_counts)}")
        return limited
    
    def _log_class_distribution(self):
        """Log class distribution."""
        counts = Counter(s['label'] for s in self.samples)
        logger.info(f"  Class distribution: {dict(counts)}")
    
    def _build_transforms(self) -> T.Compose:
        """Build image transforms."""
        if self.augment:
            return T.Compose([
                T.RandomResizedCrop(self.image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def _create_synthetic_window(self, image: torch.Tensor) -> torch.Tensor:
        """
        Create 16-frame window from single image.
        
        SPRINT-SIMPLIFICATION: Repeat with light per-frame augmentation.
        Each frame gets independent augmentation for temporal variation.
        """
        frames = []
        for i in range(self.window_size):
            if self.augment and i > 0:
                # Apply different augmentation to each frame
                # Convert to PIL for torchvision transforms
                frame_pil = TF.to_pil_image(image)
                frame = self.transform(frame_pil)
            else:
                # First frame or no augment: use base transform
                frame = image
            frames.append(frame)
        
        return torch.stack(frames)  # (16, C, H, W)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        img_path = self.base_dir / sample['path']
        image = Image.open(img_path).convert('RGB')
        
        # Apply base transform
        image = self.transform(image)  # (C, H, W)
        
        # Create synthetic 16-frame window
        window = self._create_synthetic_window(image)  # (16, C, H, W)
        
        # Label
        label = sample['label_idx']
        
        result = {
            'frames': window,           # (16, 3, 224, 224)
            'label': torch.tensor(label, dtype=torch.long),
            'sample_idx': idx,
        }
        
        # Dummy landmarks for pipeline compatibility
        if self.return_landmarks:
            # Generate canonical landmarks (same for all frames in window)
            dummy_landmarks = self._generate_dummy_landmarks()
            # Repeat for window: (16, 478, 2)
            result['landmarks'] = dummy_landmarks.unsqueeze(0).repeat(self.window_size, 1, 1)
            result['visibility'] = torch.ones(self.window_size, 478)
        
        return result
    
    def _generate_dummy_landmarks(self) -> torch.Tensor:
        """Generate canonical facial landmarks for 224x224 aligned face."""
        # Approximate MediaPipe landmark positions for frontal face
        landmarks = torch.zeros(478, 2)
        
        # Face outline (0-16)
        for i in range(17):
            angle = np.pi * (i / 16)
            landmarks[i] = torch.tensor([
                112 + 90 * np.cos(angle),
                112 + 90 * np.sin(angle)
            ])
        
        # Eyebrows, eyes, nose, mouth - simplified canonical positions
        # This is a rough approximation for pipeline compatibility
        # Real landmarks would come from MediaPipe
        
        # Left eye region (~33-42)
        for i, offset in enumerate([(33, 7, 163), (37, 7, 163), (39, 7, 163)]):
            pass  # Simplified
        
        # Just create a reasonable spread
        for i in range(478):
            if i < 17:  # jaw
                continue
            angle = 2 * np.pi * (i - 17) / (478 - 17)
            r = 50 + 30 * np.sin(angle * 3)
            landmarks[i] = torch.tensor([
                112 + r * np.cos(angle),
                112 + r * np.sin(angle)
            ])
        
        return landmarks


def get_class_weights(dataset: FER2013Dataset) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    counts = Counter(s['label'] for s in dataset.samples)
    total = len(dataset.samples)
    num_classes = len(EMOTION_CLASSES)
    
    weights = torch.zeros(num_classes)
    for cls, count in counts.items():
        weights[CLASS_TO_IDX[cls]] = total / (num_classes * count)
    
    logger.info(f"Class weights: {weights.tolist()}")
    return weights


def create_dataloaders(
    split_file: str,
    batch_size: int = 8,
    window_size: int = 16,
    image_size: int = 224,
    num_workers: int = 0,
    max_samples_per_class: Optional[int] = None,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = FER2013Dataset(
        split_file=split_file,
        split="train",
        window_size=window_size,
        image_size=image_size,
        augment=True,
        max_samples_per_class=max_samples_per_class,
        return_landmarks=True,
    )
    
    val_dataset = FER2013Dataset(
        split_file=split_file,
        split="val",
        window_size=window_size,
        image_size=image_size,
        augment=False,
        return_landmarks=True,
    )
    
    test_dataset = FER2013Dataset(
        split_file=split_file,
        split="test",
        window_size=window_size,
        image_size=image_size,
        augment=False,
        return_landmarks=True,
    )
    
    # Weighted sampler for training (handles class imbalance)
    train_sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_weights = get_class_weights(train_dataset)
        sample_weights = [class_weights[s['label_idx']] for s in train_dataset.samples]
        train_sampler = WeightedRandomSampler(
            sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        shuffle = False  # Sampler handles shuffling
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    logger.info(f"Created dataloaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    split_file = Path(__file__).parent.parent / "data" / "datasets" / "fer2013" / "processed" / "split.json"
    
    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        print("Run prepare_fer2013.py first")
        sys.exit(1)
    
    print("Testing FER2013Dataset...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        split_file=str(split_file),
        batch_size=4,
        window_size=16,
        max_samples_per_class=100,  # Small for testing
        num_workers=0,
    )
    
    # Test train batch
    batch = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"  frames: {batch['frames'].shape} (expected: B, 16, 3, 224, 224)")
    print(f"  label: {batch['label'].shape} (expected: B)")
    print(f"  landmarks: {batch['landmarks'].shape} (expected: B, 16, 478, 2)")
    print(f"  visibility: {batch['visibility'].shape} (expected: B, 16, 478)")
    print(f"  Labels in batch: {batch['label'].tolist()}")
    print(f"  Frame range: [{batch['frames'].min():.4f}, {batch['frames'].max():.4f}]")
    
    # Test val batch
    batch = next(iter(val_loader))
    print(f"\nVal batch:")
    print(f"  frames: {batch['frames'].shape}")
    print(f"  label: {batch['label'].shape}")
    
    # Test class weights
    train_dataset = FER2013Dataset(
        split_file=str(split_file),
        split="train",
        window_size=16,
        max_samples_per_class=100,
    )
    weights = get_class_weights(train_dataset)
    print(f"\nClass weights: {weights}")
    
    print("\n[OK] DataLoader test passed!")