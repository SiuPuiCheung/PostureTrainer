#!/usr/bin/env python3
"""Test script for dual model architecture."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    try:
        from src.core.base_pose_model import BasePoseModel, PoseLandmark, PoseResults
        from src.core.mediapipe_model import MediaPipePoseModel
        from src.core.model_factory import get_available_models, check_gpu_available, create_pose_model
        from src.utils.config_loader import Config
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_factory():
    """Test model factory functions."""
    print("\nTesting model factory...")
    try:
        from src.core.model_factory import get_available_models, check_gpu_available
        
        models = get_available_models()
        print(f"  Available models: {models}")
        assert "mediapipe" in models, "MediaPipe should always be available"
        
        gpu = check_gpu_available()
        print(f"  GPU available: {gpu}")
        
        print("✓ Model factory working")
        return True
    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from src.utils.config_loader import Config
        
        config = Config()
        
        # Check model config
        assert hasattr(config, 'model_config'), "Config should have model_config"
        assert 'default_type' in config.model_config, "Config should have default_type"
        assert 'use_gpu' in config.model_config, "Config should have use_gpu"
        assert 'yolov11' in config.model_config, "Config should have yolov11 settings"
        
        print(f"  Default model: {config.model_config['default_type']}")
        print(f"  Use GPU: {config.model_config['use_gpu']}")
        print(f"  YOLOv11 size: {config.model_config['yolov11']['model_size']}")
        
        print("✓ Configuration working")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_landmark():
    """Test PoseLandmark class."""
    print("\nTesting PoseLandmark...")
    try:
        from src.core.base_pose_model import PoseLandmark
        
        landmark = PoseLandmark(0.5, 0.6, 0.1, 0.9)
        assert landmark.x == 0.5
        assert landmark.y == 0.6
        assert landmark.z == 0.1
        assert landmark.visibility == 0.9
        
        print("✓ PoseLandmark working")
        return True
    except Exception as e:
        print(f"✗ PoseLandmark test failed: {e}")
        return False


def test_pose_results():
    """Test PoseResults class."""
    print("\nTesting PoseResults...")
    try:
        from src.core.base_pose_model import PoseResults, PoseLandmark
        import numpy as np
        
        landmarks = [PoseLandmark(0.1 * i, 0.2 * i, 0.0, 1.0) for i in range(33)]
        image = np.zeros((640, 480, 3), dtype=np.uint8)
        
        results = PoseResults(landmarks=landmarks, annotated_image=image)
        assert results.pose_landmarks is not None
        assert len(results.pose_landmarks) == 33
        assert results.annotated_image is not None
        
        print("✓ PoseResults working")
        return True
    except Exception as e:
        print(f"✗ PoseResults test failed: {e}")
        return False


def test_yolo_available():
    """Test if YOLOv11 dependencies are available."""
    print("\nTesting YOLOv11 availability...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        try:
            from ultralytics import YOLO
            print(f"  Ultralytics available: Yes")
            print("✓ YOLOv11 dependencies installed")
            return True
        except ImportError:
            print("  Ultralytics available: No (not installed)")
            print("⚠ YOLOv11 not available, but MediaPipe is sufficient")
            return True
    except ImportError as e:
        print(f"  PyTorch not available: {e}")
        print("⚠ YOLOv11 dependencies not installed, but MediaPipe is sufficient")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Dual Model Architecture Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_factory,
        test_config,
        test_pose_landmark,
        test_pose_results,
        test_yolo_available,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
