"""
Block 2.2: Inference Pipeline on Checkpoint
Runs full pipeline on test clip, shows predictions per segment.

SPRINT-SIMPLIFICATION: Uses fresh model (no converged checkpoint).
Full PMD: Would load trained teacher/student checkpoint.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import create_streaming_pipeline, StreamingFERDPipeline
from preprocessing.face_align import FaceAligner, FaceAlignmentResult

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

def create_test_video(output_path="test_emotion_clip.mp4", fps=10, frames_per_emotion=15):
    """Create test video from FER2013 images with known emotions."""
    print(f"Creating test video: {output_path}")
    
    base_dir = Path("data/datasets/fer2013/processed/images")
    
    # Select a few images per emotion
    emotion_samples = {}
    for emotion in EMOTIONS:
        emotion_dir = base_dir / emotion.lower()
        if emotion_dir.exists():
            images = list(emotion_dir.glob("*.jpg"))[:frames_per_emotion]
            if images:
                emotion_samples[emotion.lower()] = images
                print(f"  {emotion}: {len(images)} frames")
    
    if not emotion_samples:
        print("No images found!")
        return None
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))
    
    frame_labels = []
    
    for emotion, images in emotion_samples.items():
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, (224, 224))
                out.write(img)
                frame_labels.append(emotion)
    
    out.release()
    print(f"Created {output_path} with {len(frame_labels)} frames")
    return frame_labels

def run_inference_on_video(video_path, model, face_aligner, max_frames=None):
    """Run inference pipeline on video file."""
    print(f"\nRunning inference on {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} FPS")
    
    predictions = []
    frame_idx = 0
    
    while True:
        if max_frames and frame_idx >= max_frames:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # Face alignment
        align_result = face_aligner.align(frame)
        
        if not align_result.success:
            predictions.append({
                'frame': frame_idx,
                'pred': None,
                'conf': 0.0,
                'error': align_result.error
            })
            frame_idx += 1
            continue
        
        # Prepare tensors
        aligned = align_result.aligned_face
        tensor = torch.from_numpy(aligned).permute(2, 0, 1).unsqueeze(0).float()
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        landmarks = align_result.landmarks_224
        if landmarks is not None:
            landmarks = torch.from_numpy(landmarks).unsqueeze(0).float()
            visibility = torch.ones(1, 478)
        else:
            landmarks = torch.zeros(1, 478, 2)
            visibility = torch.ones(1, 478)
        
        # Model inference
        with torch.no_grad():
            result = model.step(tensor, landmarks, visibility)
        
        pred_idx = result['pred_class'].item()
        confidence = result['confidence'].item()
        probs = result['probs'].squeeze(0).cpu().numpy()
        
        predictions.append({
            'frame': frame_idx,
            'pred': EMOTIONS[pred_idx],
            'conf': confidence,
            'probs': probs,
            'bbox': align_result.bbox
        })
        
        # Print every 10 frames
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx:3d}: {EMOTIONS[pred_idx]:12s} ({confidence:.2%})")
        
        frame_idx += 1
    
    cap.release()
    return predictions

def analyze_predictions(predictions, frame_labels=None):
    """Analyze prediction quality."""
    print("\n=== PREDICTION ANALYSIS ===")
    
    valid_preds = [p for p in predictions if p['pred'] is not None]
    print(f"Valid predictions: {len(valid_preds)}/{len(predictions)}")
    
    if not valid_preds:
        print("No valid predictions!")
        return
    
    # Overall stats
    confidences = [p['conf'] for p in valid_preds]
    print(f"Mean confidence: {np.mean(confidences):.3f}")
    print(f"Max confidence:  {np.max(confidences):.3f}")
    print(f"Min confidence:  {np.min(confidences):.3f}")
    
    # Prediction distribution
    pred_counts = {}
    for p in valid_preds:
        pred_counts[p['pred']] = pred_counts.get(p['pred'], 0) + 1
    
    print("\nPrediction distribution:")
    for emotion, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
        print(f"  {emotion:12s}: {count:3d} ({count/len(valid_preds)*100:.1f}%)")
    
    # Compare with ground truth if available
    if frame_labels:
        print("\nGround truth comparison (per segment):")
        frames_per_emotion = len(frame_labels) // len(set(frame_labels)) if frame_labels else 15
        
        for i, emotion in enumerate(set(frame_labels) if frame_labels else []):
            start = i * frames_per_emotion
            end = min((i + 1) * frames_per_emotion, len(predictions))
            segment_preds = predictions[start:end]
            
            valid_seg = [p for p in segment_preds if p['pred'] is not None]
            if valid_seg:
                seg_pred_counts = {}
                for p in valid_seg:
                    seg_pred_counts[p['pred']] = seg_pred_counts.get(p['pred'], 0) + 1
                
                top_pred = max(seg_pred_counts.items(), key=lambda x: x[1])[0]
                match = "[OK]" if top_pred.lower() == emotion.lower() else "[X]"
                
                print(f"  Segment {emotion:12s}: Top pred = {top_pred:12s} {match} "
                      f"({seg_pred_counts.get(top_pred, 0)}/{len(valid_seg)} frames)")

def main():
    print("=" * 60)
    print("BLOCK 2.2: INFERENCE PIPELINE ON CHECKPOINT")
    print("=" * 60)
    
    # 1. Create or find test video
    test_video = "test_emotion_clip.mp4"
    if not Path(test_video).exists():
        frame_labels = create_test_video(test_video)
    else:
        print(f"Using existing {test_video}")
        # Recreate frame labels for comparison
        frame_labels = []
        base_dir = Path("data/datasets/fer2013/processed/images")
        for emotion in EMOTIONS:
            emotion_dir = base_dir / emotion.lower()
            if emotion_dir.exists():
                images = list(emotion_dir.glob("*.jpg"))[:15]
                frame_labels.extend([emotion.lower()] * len(images))
    
    # 2. Initialize pipeline with fresh model (no checkpoint)
    print("\nInitializing pipeline...")
    model = create_streaming_pipeline(
        encoder_variant='mobilevit_xs',
        encoder_pretrained=True,
        encoder_frozen=True,
        use_safm=True,
    )
    model.eval()
    print("Pipeline ready (fresh weights - no trained checkpoint)")
    
    # 3. Initialize face aligner
    print("Initializing face aligner...")
    face_aligner = FaceAligner(running_mode='VIDEO')
    
    # 4. Run inference
    predictions = run_inference_on_video(test_video, model, face_aligner, max_frames=100)
    
    # 5. Analyze
    analyze_predictions(predictions, frame_labels)
    
    # 6. Cleanup
    face_aligner.close()
    
    print("\n=== BLOCK 2.2 RESULT ===")
    print("Pipeline runs end-to-end on test clip")
    print("Predictions generated (fresh model = random-ish)")
    print("Pipeline structure validated for Block 2.3+")
    
    return predictions

if __name__ == "__main__":
    main()