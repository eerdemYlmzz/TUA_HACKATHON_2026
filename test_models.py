#!/usr/bin/env python

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from src.models import create_unet, create_autoencoder
        print("✓ Models imported successfully")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    try:
        from src.utils import calculate_psnr, calculate_ssim, plot_training_history
        print("✓ Utils imported successfully")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    return True


def test_unet_architecture():
    print("\n" + "="*70)
    print("TEST 2: U-Net Architecture")
    print("="*70)
    
    try:
        from src.models import create_unet, print_model_info
        
        model = create_unet(in_channels=3, out_channels=3, features=64)
        
        num_params = model.count_params()
        print(f"✓ U-Net instantiated with {num_params:,} trainable parameters")
        
        dummy_input = __import__('numpy').random.randn(2, 64, 64, 3).astype(__import__('numpy').float32)
        output = model(dummy_input, training=False)
        
        if output.shape == (2, 64, 64, 3):
            print(f"✓ Forward pass successful: {dummy_input.shape} → {output.shape}")
        else:
            print(f"✗ Output shape mismatch: expected (2, 64, 64, 3), got {output.shape}")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ U-Net test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autoencoder():
    print("\n" + "="*70)
    print("TEST 3: Denoising Autoencoder")
    print("="*70)
    
    try:
        from src.models import create_autoencoder
        
        model = create_autoencoder(in_channels=3, out_channels=3)
        
        num_params = model.count_params()
        print(f"✓ Autoencoder instantiated with {num_params:,} trainable parameters")
        
        dummy_input = __import__('numpy').random.randn(2, 64, 64, 3).astype(__import__('numpy').float32)
        output = model(dummy_input, training=False)
        
        if output.shape == (2, 64, 64, 3):
            print(f"✓ Forward pass successful: {dummy_input.shape} → {output.shape}")
        else:
            print(f"✗ Output shape mismatch: expected (2, 64, 64, 3), got {output.shape}")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Autoencoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    print("\n" + "="*70)
    print("TEST 4: Metrics Calculation")
    print("="*70)
    
    try:
        import numpy as np
        from src.utils import calculate_psnr, calculate_ssim, calculate_mse
        
        original = np.random.rand(64, 64, 3).astype(np.float32)
        noisy = original + np.random.randn(64, 64, 3).astype(np.float32) * 0.05
        noisy = np.clip(noisy, 0, 1)
        
        psnr = calculate_psnr(original, noisy)
        ssim = calculate_ssim(original, noisy)
        mse = calculate_mse(original, noisy)
        
        if psnr > 0 and 0 <= ssim <= 1 and mse >= 0:
            print(f"✓ PSNR calculated: {psnr:.2f} dB")
            print(f"✓ SSIM calculated: {ssim:.4f}")
            print(f"✓ MSE calculated: {mse:.6f}")
            return True
        else:
            print(f"✗ Metric values out of expected range")
            print(f"  PSNR: {psnr}, SSIM: {ssim}, MSE: {mse}")
            return False
    
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    print("\n" + "="*70)
    print("TEST 5: Loss Functions")
    print("="*70)
    
    try:
        import tensorflow as tf
        from src.utils import CombinedLoss
        
        output = tf.constant(__import__('numpy').random.randn(2, 64, 64, 3).astype(__import__('numpy').float32))
        target = tf.constant(__import__('numpy').random.randn(2, 64, 64, 3).astype(__import__('numpy').float32))
        
        mse_loss = tf.keras.losses.MeanSquaredError()
        loss_mse = mse_loss(output, target).numpy()
        print(f"✓ MSELoss: {loss_mse:.6f}")
        
        l1_loss = tf.keras.losses.MeanAbsoluteError()
        loss_l1 = l1_loss(output, target).numpy()
        print(f"✓ L1Loss: {loss_l1:.6f}")
        
        combined_loss = CombinedLoss(weight_l1=0.5, weight_mse=0.5)
        loss_combined = combined_loss(output, target).numpy()
        print(f"✓ CombinedLoss: {loss_combined:.6f}")
        
        return True
    
    except Exception as e:
        print(f"✗ Loss functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tensor_conversions():
    print("\n" + "="*70)
    print("TEST 6: Tensor Conversions")
    print("="*70)
    
    try:
        import tensorflow as tf
        import numpy as np
        from src.utils import tensor_to_numpy, numpy_to_tensor
        
        np_array = np.random.rand(3, 64, 64).astype(np.float32)
        tensor = numpy_to_tensor(np_array)
        
        if isinstance(tensor, tf.Tensor):
            print(f"✓ NumPy → Tensor: {np_array.shape} → {tensor.shape}")
        else:
            print(f"✗ Conversion failed: output is not a tensor")
            return False
        
        result = tensor_to_numpy(tensor)
        
        if isinstance(result, np.ndarray) and result.shape == np_array.shape:
            print(f"✓ Tensor → NumPy: {tuple(tensor.shape)} → {result.shape}")
        else:
            print(f"✗ Conversion failed: output shape/type mismatch")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Tensor conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_processing():
    print("\n" + "="*70)
    print("TEST 7: Image Processing Utils")
    print("="*70)
    
    try:
        import numpy as np
        from src.utils import normalize_image, clip_image
        
        image = np.random.rand(64, 64, 3) * 255
        
        normalized = normalize_image(image, max_value=1.0)
        
        if normalized.min() >= 0 and normalized.max() <= 1:
            print(f"✓ Image normalized: {image.min():.2f}-{image.max():.2f} → {normalized.min():.2f}-{normalized.max():.2f}")
        else:
            print(f"✗ Normalization failed: values out of range")
            return False
        
        clipped = clip_image(normalized + 0.5, min_val=0, max_val=1)
        
        if clipped.min() >= 0 and clipped.max() <= 1:
            print(f"✓ Image clipped: values in [0, 1]")
        else:
            print(f"✗ Clipping failed: values out of range")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Image processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load():
    print("\n" + "="*70)
    print("TEST 8: Model Save/Load")
    print("="*70)
    
    try:
        import tempfile
        from pathlib import Path
        from src.models import create_unet
        from src.utils import save_model, load_model
        
        model = create_unet(in_channels=3, out_channels=3, features=64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model.h5"
            save_model(model, str(save_path), epoch=5, metrics={'test': 0.5})
            
            if save_path.exists():
                print(f"✓ Model saved successfully ({save_path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"✗ Model save failed: file not created")
                return False
            
            new_model = create_unet(in_channels=3, out_channels=3, features=64)
            metadata = load_model(new_model, str(save_path))
            
            if metadata['epoch'] == 5:
                print(f"✓ Model loaded successfully (epoch: {metadata['epoch']})")
            else:
                print(f"✗ Model load failed: metadata mismatch")
                return False
        
        return True
    
    except Exception as e:
        print(f"✗ Model save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "█"*70)
    print("TUA HACKATHON 2026 - DENOISING PROJECT VERIFICATION")
    print("█"*70)
    
    tests = [
        ("Module Imports", test_imports),
        ("U-Net Architecture", test_unet_architecture),
        ("Denoising Autoencoder", test_autoencoder),
        ("Metrics Calculation", test_metrics),
        ("Loss Functions", test_loss_functions),
        ("Tensor Conversions", test_tensor_conversions),
        ("Image Processing", test_image_processing),
        ("Model Save/Load", test_model_save_load),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Project is ready for integration.\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
