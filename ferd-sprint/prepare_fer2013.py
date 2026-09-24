#!/usr/bin/env python
"""
Prepare FER2013 dataset for FERD sprint from kagglehub download.
The dataset is already organized in train/test folders with class subdirectories.
"""
import os
import sys
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm

SOURCE_DIR = Path(r"C:\Users\Monika\.cache\kagglehub\datasets\msambare\fer2013\versions\1")
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

LABEL_TO_IDX = {v: k for k, v in EMOTION_LABELS.items()}

def copy_dataset():
    """Copy train/test data to processed directory and create train/val split."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    for label_name in EMOTION_LABELS.values():
        (IMAGES_DIR / label_name).mkdir(exist_ok=True)
    
    all_samples = []
    
    # Process train folder
    train_dir = SOURCE_DIR / "train"
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            if label not in EMOTION_LABELS.values():
                continue
            for img_path in class_dir.glob("*.jpg"):
                all_samples.append({
                    "path": f"images/{label}/{img_path.name}",
                    "label": label,
                    "label_idx": LABEL_TO_IDX[label],
                    "source": str(img_path),
                    "split": "train"
                })
    
    # Process test folder
    test_dir = SOURCE_DIR / "test"
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            if label not in EMOTION_LABELS.values():
                continue
            for img_path in class_dir.glob("*.jpg"):
                all_samples.append({
                    "path": f"images/{label}/{img_path.name}",
                    "label": label,
                    "label_idx": LABEL_TO_IDX[label],
                    "source": str(img_path),
                    "split": "test"
                })
    
    print(f"Total samples found: {len(all_samples)}")
    
    # Copy all images to processed/images/
    print("Copying images...")
    for sample in tqdm(all_samples, desc="Copying"):
        dest_path = PROCESSED_DIR / sample["path"]
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["source"], dest_path)
    
    # Create train/val split: use 85% of train for training, 15% for validation
    # test set can be kept as held-out test
    train_samples = [s for s in all_samples if s["split"] == "train"]
    test_samples = [s for s in all_samples if s["split"] == "test"]
    
    random.seed(42)
    random.shuffle(train_samples)
    
    val_size = int(len(train_samples) * 0.15)
    val_samples = train_samples[:val_size]
    final_train_samples = train_samples[val_size:]
    
    # Update split labels
    for s in final_train_samples:
        s["split"] = "train"
    for s in val_samples:
        s["split"] = "val"
    for s in test_samples:
        s["split"] = "test"
    
    # Prepare final sample list (without source paths)
    final_samples = []
    for s in final_train_samples + val_samples + test_samples:
        final_samples.append({
            "path": s["path"],
            "label": s["label"],
            "label_idx": s["label_idx"],
            "split": s["split"]
        })
    
    # Save split file
    split_path = PROCESSED_DIR / "split.json"
    with open(split_path, 'w') as f:
        json.dump(final_samples, f, indent=2)
    
    # Print statistics
    train_count = len(final_train_samples)
    val_count = len(val_samples)
    test_count = len(test_samples)
    
    print(f"\nSplit created:")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")
    print(f"  Total: {train_count + val_count + test_count}")
    
    # Class distribution
    for split_name, split_samples in [("train", final_train_samples), ("val", val_samples), ("test", test_samples)]:
        print(f"\n  {split_name.capitalize()} class distribution:")
        for label in EMOTION_LABELS.values():
            count = sum(1 for s in split_samples if s["label"] == label)
            print(f"    {label}: {count}")
    
    # Write class mapping
    mapping_path = PROCESSED_DIR / "class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(LABEL_TO_IDX, f, indent=2)
    
    return train_count, val_count, test_count

def main():
    print("=" * 60)
    print("FER2013 Dataset Preparation for FERD Sprint")
    print("=" * 60)
    
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Run download_fer2013.py first (or ensure kagglehub downloaded it)")
        sys.exit(1)
    
    train_count, val_count, test_count = copy_dataset()
    
    total = train_count + val_count + test_count
    if total >= 3000:
        print(f"\n✅ SUCCESS: {total} images prepared (≥3000 required)")
    else:
        print(f"\n⚠️  WARNING: Only {total} images (target ≥3000)")
    
    print(f"\nDataset location: {PROCESSED_DIR}")
    print("Structure:")
    print(f"  {PROCESSED_DIR}/")
    print(f"    images/")
    for label in EMOTION_LABELS.values():
        label_dir = IMAGES_DIR / label
        count = len(list(label_dir.glob("*.jpg"))) if label_dir.exists() else 0
        print(f"      {label}/ ({count} images)")
    print(f"    split.json")
    print(f"    class_mapping.json")

if __name__ == "__main__":
    main()