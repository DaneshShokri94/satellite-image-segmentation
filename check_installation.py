#!/usr/bin/env python3
"""
Installation Verification Script
================================
Verify all dependencies are installed correctly.

Usage:
    python check_installation.py
"""

import sys

def check_python_version():
    """Check Python version."""
    print("=" * 50)
    print("PYTHON VERSION")
    print("=" * 50)
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        return False
    else:
        print("✓ Python version OK")
        return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\n" + "=" * 50)
    print("PYTORCH")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print("✓ PyTorch installed")
        
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"         Memory: {mem:.1f} GB")
            print("✓ CUDA OK")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
        
        if hasattr(torch.backends, 'mps'):
            print(f"\nMPS available: {torch.backends.mps.is_available()}")
            if torch.backends.mps.is_available():
                print("✓ Apple Silicon GPU OK")
        
        print("\nTesting tensor operations...")
        x = torch.rand(3, 3)
        y = torch.rand(3, 3)
        z = torch.matmul(x, y)
        print("✓ CPU tensor operations OK")
        
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            z = torch.matmul(x, y)
            print("✓ CUDA tensor operations OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch not installed: {e}")
        print("\nInstall PyTorch:")
        print("  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
        return False
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False


def check_torchvision():
    """Check torchvision installation."""
    print("\n" + "=" * 50)
    print("TORCHVISION")
    print("=" * 50)
    
    try:
        import torchvision
        print(f"torchvision version: {torchvision.__version__}")
        print("✓ torchvision installed")
        return True
    except ImportError as e:
        print(f"❌ torchvision not installed: {e}")
        return False


def check_smp():
    """Check segmentation_models_pytorch installation."""
    print("\n" + "=" * 50)
    print("SEGMENTATION MODELS PYTORCH")
    print("=" * 50)
    
    try:
        import segmentation_models_pytorch as smp
        print(f"smp version: {smp.__version__}")
        print("✓ segmentation_models_pytorch installed")
        
        print("\nTesting model creation...")
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, classes=1)
        print("✓ Model creation OK")
        
        print(f"\nAvailable encoders: {len(smp.encoders.get_encoder_names())}")
        
        return True
    except ImportError as e:
        print(f"❌ segmentation_models_pytorch not installed: {e}")
        print("\nInstall with:")
        print("  pip install segmentation-models-pytorch")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_other_dependencies():
    """Check other dependencies."""
    print("\n" + "=" * 50)
    print("OTHER DEPENDENCIES")
    print("=" * 50)
    
    dependencies = {
        'PIL': 'pillow',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
    }
    
    optional_deps = {
        'matplotlib': 'matplotlib',
        'rasterio': 'rasterio',
    }
    
    all_ok = True
    
    print("\nRequired:")
    for module, package in dependencies.items():
        try:
            if module == 'PIL':
                from PIL import Image
                import PIL
                print(f"  ✓ {module} ({PIL.__version__})")
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                print(f"  ✓ {module} ({version})")
        except ImportError:
            print(f"  ❌ {module} - install with: pip install {package}")
            all_ok = False
    
    print("\nOptional:")
    for module, package in optional_deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {module} ({version})")
        except ImportError:
            print(f"  ⚠ {module} not installed (optional) - pip install {package}")
    
    return all_ok


def test_full_pipeline():
    """Test the full pipeline with a dummy forward pass."""
    print("\n" + "=" * 50)
    print("FULL PIPELINE TEST")
    print("=" * 50)
    
    try:
        import torch
        import segmentation_models_pytorch as smp
        
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 256, 256)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass OK")
        
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            with torch.no_grad():
                output = model(x)
            print("✓ GPU forward pass OK")
        
        # Test multi-class
        model_multi = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=5,
        )
        x = torch.randn(2, 3, 256, 256)
        model_multi.eval()
        with torch.no_grad():
            output = model_multi(x)
        print(f"Multi-class output shape: {output.shape}")
        print("✓ Multi-class model OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def main():
    print("\n" + "=" * 50)
    print(" SATELLITE IMAGE SEGMENTATION - INSTALLATION CHECK")
    print("=" * 50)
    
    results = []
    
    results.append(("Python", check_python_version()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("torchvision", check_torchvision()))
    results.append(("SMP", check_smp()))
    results.append(("Dependencies", check_other_dependencies()))
    results.append(("Pipeline", test_full_pipeline()))
    
    print("\n" + "=" * 50)
    print(" SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print(" ✓ ALL CHECKS PASSED - Ready to train!")
    else:
        print(" ❌ SOME CHECKS FAILED - Fix issues above")
    print("=" * 50 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
