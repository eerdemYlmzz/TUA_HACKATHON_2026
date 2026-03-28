import tensorflow as tf
import numpy as np
import yaml
import cv2
from pathlib import Path

from src.models import create_unet, print_model_info
from src.utils import (
    load_model, plot_denoising_result, plot_training_history,
    calculate_psnr, calculate_ssim, clip_image
)
from src.preprocessing import select_images, load_and_add_noise


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main testing pipeline for trained model."""
    config = load_config("config.yaml")
    
    print("\n" + "="*60)
    print("Loading trained model...")
    print("="*60)
    
    # Load trained model
    model_path = config['paths']['models'] + "/denoiser_v1.h5"
    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run 'python main.py' first to train the model")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        print_model_info(model, "UNET")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Load test data
    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)
    
    try:
        raw_data_path = config['paths']['raw_data']
        
        if not Path(raw_data_path).exists():
            print(f"ERROR: Raw data path not found: {raw_data_path}")
            return
        
        # Select images
        image_paths = select_images(raw_data_path, n_samples=200)
        print(f"Found {len(image_paths)} images for testing")
        
        if len(image_paths) == 0:
            print("ERROR: No images found!")
            return
        
        # Load images
        original_images = []
        noisy_images = []
        
        noise_type = config['params'].get('noise_type', 'salt_and_pepper')
        
        for idx, img_path in enumerate(image_paths):
            if idx % 40 == 0:
                print(f"Loading: {idx}/{len(image_paths)}")
            
            try:
                original, noisy = load_and_add_noise(img_path, noise_type=noise_type)
                
                if original is not None:
                    # Resize to model input size (64x64)
                    original = cv2.resize(original, (64, 64))
                    noisy = cv2.resize(noisy, (64, 64))
                    
                    original_images.append(original)
                    noisy_images.append(noisy)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        print(f"Loaded {len(original_images)} images successfully")
        
        if len(original_images) == 0:
            print("ERROR: Could not load any valid images!")
            return
        
        # Convert to numpy arrays
        X_data = np.array(noisy_images, dtype=np.float32)
        y_data = np.array(original_images, dtype=np.float32)
        
        # Use 70% for testing (since we want to test on diverse data)
        total_samples = len(X_data)
        test_size = int(total_samples * 0.7)
        
        # Shuffle indices
        indices = np.random.permutation(total_samples)
        test_indices = indices[:test_size]
        
        X_test = X_data[test_indices]
        y_test = y_data[test_indices]
        
        print(f"\nTest set: {len(X_test)} images")
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate model on test set
    print("\n" + "="*60)
    print("Evaluating model on test set...")
    print("="*60)
    
    try:
        test_psnr_scores = []
        test_ssim_scores = []
        test_mse_scores = []
        
        # Process each test sample
        for i in range(len(X_test)):
            if i % 20 == 0:
                print(f"Processing: {i}/{len(X_test)}")
            
            try:
                # Denoise
                denoised = model.predict(X_test[i:i+1], verbose=0)[0]
                denoised = clip_image(denoised)
                
                # Calculate metrics
                psnr = calculate_psnr(y_test[i], denoised)
                ssim = calculate_ssim(y_test[i], denoised)
                mse = np.mean((y_test[i] - denoised) ** 2)
                
                test_psnr_scores.append(psnr)
                test_ssim_scores.append(ssim)
                test_mse_scores.append(mse)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate averages
        if len(test_psnr_scores) > 0:
            avg_psnr = np.mean(test_psnr_scores)
            std_psnr = np.std(test_psnr_scores)
            
            avg_ssim = np.mean(test_ssim_scores)
            std_ssim = np.std(test_ssim_scores)
            
            avg_mse = np.mean(test_mse_scores)
            std_mse = np.std(test_mse_scores)
            
            print("\n" + "="*60)
            print("TEST SET EVALUATION RESULTS")
            print("="*60)
            print(f"Samples evaluated: {len(test_psnr_scores)}")
            print(f"\nPeak Signal-to-Noise Ratio (PSNR):")
            print(f"  Average: {avg_psnr:.4f} dB")
            print(f"  Std Dev: {std_psnr:.4f} dB")
            print(f"  Min: {np.min(test_psnr_scores):.4f} dB")
            print(f"  Max: {np.max(test_psnr_scores):.4f} dB")
            
            print(f"\nStructural Similarity Index (SSIM):")
            print(f"  Average: {avg_ssim:.4f}")
            print(f"  Std Dev: {std_ssim:.4f}")
            print(f"  Min: {np.min(test_ssim_scores):.4f}")
            print(f"  Max: {np.max(test_ssim_scores):.4f}")
            
            print(f"\nMean Squared Error (MSE):")
            print(f"  Average: {avg_mse:.6f}")
            print(f"  Std Dev: {std_mse:.6f}")
            print(f"  Min: {np.min(test_mse_scores):.6f}")
            print(f"  Max: {np.max(test_mse_scores):.6f}")
            
            # Save detailed results
            results_file = Path(config['paths']['outputs']) / "test_results.txt"
            with open(results_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write("TEST SET EVALUATION RESULTS\n")
                f.write("="*60 + "\n")
                f.write(f"Samples evaluated: {len(test_psnr_scores)}\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Noise type: {noise_type}\n\n")
                
                f.write("Peak Signal-to-Noise Ratio (PSNR):\n")
                f.write(f"  Average: {avg_psnr:.4f} dB\n")
                f.write(f"  Std Dev: {std_psnr:.4f} dB\n")
                f.write(f"  Min: {np.min(test_psnr_scores):.4f} dB\n")
                f.write(f"  Max: {np.max(test_psnr_scores):.4f} dB\n\n")
                
                f.write("Structural Similarity Index (SSIM):\n")
                f.write(f"  Average: {avg_ssim:.4f}\n")
                f.write(f"  Std Dev: {std_ssim:.4f}\n")
                f.write(f"  Min: {np.min(test_ssim_scores):.4f}\n")
                f.write(f"  Max: {np.max(test_ssim_scores):.4f}\n\n")
                
                f.write("Mean Squared Error (MSE):\n")
                f.write(f"  Average: {avg_mse:.6f}\n")
                f.write(f"  Std Dev: {std_mse:.6f}\n")
                f.write(f"  Min: {np.min(test_mse_scores):.6f}\n")
                f.write(f"  Max: {np.max(test_mse_scores):.6f}\n")
            
            print(f"\n✅ Results saved to: {results_file}")
        
        else:
            print("ERROR: No test samples processed!")
            return
    
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save sample test denoising results
    print("\n" + "="*60)
    print("Saving sample denoising results...")
    print("="*60)
    
    try:
        for i in range(min(5, len(X_test))):
            denoised = model.predict(X_test[i:i+1], verbose=0)[0]
            denoised = clip_image(denoised)
            
            plot_denoising_result(
                X_test[i],
                denoised,
                y_test[i],
                save_path=str(Path(config['paths']['outputs']) / f"test_sample_{i}.png")
            )
            
            psnr = calculate_psnr(y_test[i], denoised)
            ssim = calculate_ssim(y_test[i], denoised)
            print(f"Sample {i}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    except Exception as e:
        print(f"ERROR saving results: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with different noise levels
    print("\n" + "="*60)
    print("Testing with different noise levels...")
    print("="*60)
    
    try:
        from src.noises import salt_and_pepper
        
        noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        print(f"\nNoise Type: salt_and_pepper")
        print(f"{'Noise Level':<15} {'PSNR (dB)':<15} {'SSIM':<15} {'MSE':<15}")
        print("-" * 60)
        
        for noise_level in noise_levels:
            psnr_scores = []
            ssim_scores = []
            mse_scores = []
            
            # Test on subset of data
            test_subset_size = min(20, len(X_test))
            
            for i in range(test_subset_size):
                # Add noise with specific level
                noisy = salt_and_pepper(y_test[i].copy(), noise_level=noise_level)
                
                # Denoise
                denoised = model.predict(np.expand_dims(noisy, 0), verbose=0)[0]
                denoised = clip_image(denoised)
                
                # Calculate metrics
                psnr = calculate_psnr(y_test[i], denoised)
                ssim = calculate_ssim(y_test[i], denoised)
                mse = np.mean((y_test[i] - denoised) ** 2)
                
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
                mse_scores.append(mse)
            
            avg_psnr = np.mean(psnr_scores)
            avg_ssim = np.mean(ssim_scores)
            avg_mse = np.mean(mse_scores)
            
            print(f"{noise_level:<15.2f} {avg_psnr:<15.2f} {avg_ssim:<15.4f} {avg_mse:<15.6f}")
    
    except Exception as e:
        print(f"ERROR testing noise levels: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Model testing completed successfully!")
    print("="*60)
    print(f"\nResults saved to: {config['paths']['outputs']}")


if __name__ == "__main__":
    main()
