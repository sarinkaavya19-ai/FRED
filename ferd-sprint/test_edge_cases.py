"""
Block 2.5: Bug Bash & Edge-Case Hardening
Tests pipeline against PMD Section 8.1 edge cases:
1. No face in frame
2. Extreme pose (>60° yaw)
3. Low light
4. Brief occlusion (hand over face)

Goal: Ensure no crashes, graceful degradation acceptable.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import create_streaming_pipeline, StreamingFERDPipeline
from preprocessing.face_align import FaceAligner, FaceAlignmentResult

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]


def create_test_images():
    """Create synthetic test images for edge cases."""
    test_dir = Path("edge_case_tests")
    test_dir.mkdir(exist_ok=True)
    
    # 1. Blank frame (no face)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "blank_frame.jpg"), blank)
    
    # 2. Low light frame (dark)
    low_light = np.ones((480, 640, 3), dtype=np.uint8) * 30
    cv2.imwrite(str(test_dir / "low_light.jpg"), low_light)
    
    # 3. Very bright frame (overexposed)
    bright = np.ones((480, 640, 3), dtype=np.uint8) * 240
    cv2.imwrite(str(test_dir / "overexposed.jpg"), bright)
    
    # 4. Extreme pose simulation - face at edge
    edge_face = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Draw a simple face-like shape at left edge (simulating extreme yaw)
    cv2.ellipse(edge_face, (100, 240), (80, 100), 0, 0, 360, (100, 100, 100), -1)
    cv2.circle(edge_face, (70, 220), 10, (50, 50, 50), -1)  # left eye
    cv2.circle(edge_face, (130, 220), 10, (50, 50, 50), -1)  # right eye
    cv2.ellipse(edge_face, (100, 280), (20, 10), 0, 0, 180, (50, 50, 50), 2)  # mouth
    cv2.imwrite(str(test_dir / "extreme_pose_left.jpg"), edge_face)
    
    # 5. Occlusion simulation - rectangle over mouth region
    occ = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.ellipse(occ, (320, 240), (80, 100), 0, 0, 360, (100, 100, 100), -1)
    cv2.circle(occ, (280, 220), 10, (50, 50, 50), -1)
    cv2.circle(occ, (360, 220), 10, (50, 50, 50), -1)
    cv2.ellipse(occ, (320, 280), (20, 10), 0, 0, 180, (50, 50, 50), 2)
    # Hand over mouth (rectangle)
    cv2.rectangle(occ, (280, 260), (360, 300), (150, 100, 50), -1)
    cv2.imwrite(str(test_dir / "occlusion_mouth.jpg"), occ)
    
    # 6. Normal face for baseline
    normal = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.ellipse(normal, (320, 240), (80, 100), 0, 0, 360, (100, 100, 100), -1)
    cv2.circle(normal, (280, 220), 10, (50, 50, 50), -1)
    cv2.circle(normal, (360, 220), 10, (50, 50, 50), -1)
    cv2.ellipse(normal, (320, 280), (20, 10), 0, 0, 180, (50, 50, 50), 2)
    cv2.imwrite(str(test_dir / "normal_face.jpg"), normal)
    
    print(f"Created test images in {test_dir}/")
    return test_dir


def test_edge_case(name, frame, model, face_aligner):
    """Test a single edge case frame."""
    print(f"\nTesting: {name}")
    print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")
    
    try:
        # Face alignment
        align_result = face_aligner.align(frame)
        print(f"  Alignment: {'OK' if align_result.success else 'FAILED - ' + str(align_result.error)}")
        
        if not align_result.success:
            # This is expected for some edge cases - should not crash
            print(f"  Result: Graceful degradation (no face detected)")
            return True
        
        # Prepare tensors
        aligned = align_result.aligned_face
        tensor = torch.from_numpy(aligned).permute(2, 0, 1).unsqueeze(0).float()
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
        
        print(f"  Prediction: {EMOTIONS[pred_idx]} ({confidence:.1%})")
        print(f"  Result: OK")
        return True
        
    except Exception as e:
        print(f"  CRASH: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("BLOCK 2.5: BUG BASH & EDGE-CASE HARDENING")
    print("=" * 60)
    
    # Create test images
    test_dir = create_test_images()
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    model = create_streaming_pipeline(
        encoder_variant='mobilevit_xs',
        encoder_pretrained=True,
        encoder_frozen=True,
        use_safm=True,
    )
    model.eval()
    
    print("Initializing face aligner...")
    face_aligner = FaceAligner(running_mode='IMAGE')  # Use IMAGE mode for static tests
    
    # Test cases
    test_cases = [
        ("Blank frame (no face)", cv2.imread(str(test_dir / "blank_frame.jpg"))),
        ("Low light", cv2.imread(str(test_dir / "low_light.jpg"))),
        ("Overexposed", cv2.imread(str(test_dir / "overexposed.jpg"))),
        ("Extreme pose (left)", cv2.imread(str(test_dir / "extreme_pose_left.jpg"))),
        ("Occlusion (mouth)", cv2.imread(str(test_dir / "occlusion_mouth.jpg"))),
        ("Normal face (baseline)", cv2.imread(str(test_dir / "normal_face.jpg"))),
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING EDGE CASE TESTS")
    print("=" * 60)
    
    results = {}
    for name, frame in test_cases:
        if frame is None:
            print(f"\nSkipping {name}: image not found")
            results[name] = False
            continue
        results[name] = test_edge_case(name, frame, model, face_aligner)
    
    # Summary
    print("\n" + "=" * 60)
    print("EDGE CASE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL (CRASH)"
        print(f"  {name:30s}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n[OK] ALL EDGE CASES HANDLED - NO CRASHES")
    else:
        print(f"\n⚠️  {total - passed} EDGE CASE(S) CAUSED CRASHES - NEED FIXES")
    
    # Cleanup
    face_aligner.close()
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)