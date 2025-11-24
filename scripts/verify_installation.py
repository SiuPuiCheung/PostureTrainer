"""Quick smoke test to verify installation."""

import sys

def check_imports():
    """Check that all required packages can be imported."""
    print("Checking imports...")
    
    packages = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('yaml', 'pyyaml'),
        ('ultralytics', 'ultralytics'),
    ]
    
    missing = []
    for module, package in packages:
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All required packages installed!")
    return True


def check_project_structure():
    """Check that project structure is correct."""
    print("\nChecking project structure...")
    
    import os
    from pathlib import Path
    
    project_root = Path(__file__).parent
    required_paths = [
        'src/core/pose_analysis.py',
        'src/core/pose_detection.py',
        'src/utils/config_loader.py',
        'src/utils/geometry.py',
        'src/utils/visualization.py',
        'src/utils/report.py',
        'src/data/capture.py',
        'src/models/pose_estimators.py',
        'config/config.yaml',
        'main.py',
    ]
    
    missing = []
    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"✓ {path}")
        else:
            print(f"✗ {path} (missing)")
            missing.append(path)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    print("\n✅ Project structure is correct!")
    return True


def run_config_test():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config_loader import Config
        config = Config()
        
        print(f"✓ Config loaded")
        print(f"✓ Found {len(config.analysis_types)} analysis types")
        print(f"✓ Output directory: {config.paths['output_dir']}")
        
        print("\n✅ Configuration test passed!")
        return True
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Posture Trainer - Installation Verification")
    print("=" * 60)
    
    checks = [
        check_imports(),
        check_project_structure(),
        run_config_test(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All checks passed! Ready to run: python main.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
