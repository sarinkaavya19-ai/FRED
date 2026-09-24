#!/usr/bin/env python
"""
Download and prepare FER2013 dataset for FERD sprint using kagglehub.
FER2013: 35,887 grayscale 48x48 face images, 7 emotion classes.
Classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
"""
import os
import sys
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import kagglehub

DATA_DIR = Path(__file__).parent / "data" / "datasets" / "fer2013"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"

EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

def download_fer2013():
    """Download FER2013 dataset using kagglehub."""
    print("Downloading FER2013 dataset via kagglehub...")
    path = kagglehub.dataset_download("msambare/fer2013")
    print(f"Dataset downloaded to: {path}")
    
    # Find the CSV file
    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset")
    
    csv_path = csv_files[0]
    print(f"Found CSV: {csv_path}")
    return csv_path

def process_fer2013(csv_path, max_per_class=None):
    """Convert CSV to individual image files organized by class."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    for label_name in EMOTION_LABELS.values():
        (IMAGES_DIR / label_name).mkdir(exist_ok=True)
    
    class_counts = {label: 0 for label in EMOTION_LABELS.values()}
    
    print("Processing FER2013 CSV...")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in tqdm(rows, desc="Converting images"):
        emotion_idx = int(row['emotion'])
        emotion_name = EMOTION_LABELS[emotion_idx]
        
        if max_per_class and class_counts[emotion_name] >= max_per_class:
            continue
        
        # Parse pixel data
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
        img_array = pixels.reshape(48, 48)
        
        # Save as PNG
        img = Image.fromarray(img_array, mode='L')
        save_path = IMAGES_DIR / emotion_name / f"{emotion_name}_{class_counts[emotion_name]:05d}.png"
        img.save(save_path)
        
        class_counts[emotion_name] += 1
    
    print("\nDataset statistics:")
    total = 0
    for label, count in class_counts.items():
        print(f"  {label}: {count} images")
        total += count
    print(f"  Total: {total} images")
    
    # Write class mapping for reference
    import json
    mapping_path = PROCESSED_DIR / "class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump({v: k for k, v in EMOTION_LABELS.items()}, f, indent=2)
    
    return class_counts

def create_train_val_split(processed_dir, val_ratio=0.15):
    """Create train/val split files."""
    import json
    import random
    
    all_samples = []
    for class_dir in (processed_dir / "images").iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.glob("*.png"):
                all_samples.append({
                    "path": str(img_path.relative_to(processed_dir)),
                    "label": class_dir.name,
                    "label_idx": {v: k for k, v in EMOTION_LABELS.items()}[class_dir.name]
                })
    
    random.seed(42)
    random.shuffle(all_samples)
    
    val_size = int(len(all_samples) * val_ratio)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    
    split_path = processed_dir / "split.json"
    with open(split_path, 'w') as f:
        json.dump({"train": train_samples, "val": val_samples}, f, indent=2)
    
    print(f"\nTrain/Val split created: {len(train_samples)} train, {len(val_samples)} val")
    return split_path

def main():
    print("=" * 60)
    print("FER2013 Dataset Download & Preparation for FERD Sprint")
    print("=" * 60)
    
    csv_path = download_fer2013()
    class_counts = process_fer2013(csv_path, max_per_class=1000)  # Limit for speed
    create_train_val_split(PROCESSED_DIR)
    
    total = sum(class_counts.values())
    if total >= 3000:
        print(f"\n✅ SUCCESS: {total} images downloaded and processed (≥3000 required)")
    else:
        print(f"\n⚠️  WARNING: Only {total} images (target ≥3000)")
    
    print(f"\nDataset location: {PROCESSED_DIR}")
    print("Structure:")
    print(f"  {PROCESSED_DIR}/")
    print(f"    images/")
    for label in EMOTION_LABELS.values():
        count = class_counts.get(label, 0)
        print(f"      {label}/ ({count} images)")
    print(f"    split.json")
    print(f"    class_mapping.json")

if __name__ == "__main__":
    main()