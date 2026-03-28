import tensorflow as tf
import numpy as np
import yaml
import cv2
from pathlib import Path

from src.models import create_unet, create_autoencoder, print_model_info
from src.utils import (
    save_model, load_model, CombinedLoss,
    plot_denoising_result, plot_training_history,
    calculate_psnr, calculate_ssim, clip_image,
    normalize_image, tensor_to_numpy, numpy_to_tensor
)


def load_config(config_path: str = "config.yaml") -> dict:
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_model(config: dict):
    
    model_config = config['model']
    architecture = model_config['architecture']
    
    if architecture == "unet":
        unet_params = model_config['unet']
        model = create_unet(
            in_channels=unet_params['in_channels'],
            out_channels=unet_params['out_channels'],
            features=unet_params['features']
        )
    elif architecture == "autoencoder":
        ae_params = model_config['autoencoder']
        model = create_autoencoder(
            in_channels=ae_params['in_channels'],
            out_channels=ae_params['out_channels']
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def setup_training(config: dict, model):
    
    training_config = config['training']
    
    if training_config['optimizer'] == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_config['learning_rate']
        )
    elif training_config['optimizer'] == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=training_config['learning_rate'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
    
    if training_config['loss_function'] == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()
    elif training_config['loss_function'] == "l1":
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif training_config['loss_function'] == "combined":
        from src.utils import CombinedLoss
        loss_fn = CombinedLoss(weight_l1=0.5, weight_mse=0.5)
    else:
        raise ValueError(f"Unknown loss function: {training_config['loss_function']}")
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mse'])
    
    return model, optimizer, loss_fn


def main():
    
    config = load_config("config.yaml")
    
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    model = initialize_model(config)
    print_model_info(model, config['model']['architecture'].upper())
    
    model, optimizer, loss_fn = setup_training(config, model)
    print(f"Optimizer: {config['training']['optimizer'].upper()}")
    print(f"Loss function: {config['training']['loss_function'].upper()}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    
    # Create output directories
    Path(config['paths']['outputs']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['models']).mkdir(parents=True, exist_ok=True)
    
    print("\nSetup complete! Model is ready for training.")
    
    # Load training data
    print("\n" + "="*60)
    print("Loading training data...")
    print("="*60)
    
    try:
        from src.preprocessing import select_images, load_and_add_noise
        
        raw_data_path = config['paths']['raw_data']
        
        # Check if raw data exists
        if not Path(raw_data_path).exists():
            print(f"WARNING: Raw data path '{raw_data_path}' not found!")
            print("Please ensure your training data is in:", raw_data_path)
            print("Exiting without training...")
            return
        
        # Select and load images
        image_paths = select_images(raw_data_path, n_samples=100)
        print(f"Found {len(image_paths)} images for training")
        
        if len(image_paths) == 0:
            print("ERROR: No images found in data directory!")
            return
        
        # Prepare training dataset
        original_images = []
        noisy_images = []
        
        for idx, img_path in enumerate(image_paths):
            if idx % 20 == 0:
                print(f"Loading: {idx}/{len(image_paths)}")
            
            original, noisy = load_and_add_noise(
                img_path, 
                noise_type=config['params'].get('noise_type', 'salt_and_pepper')
            )
            
            if original is not None:
                # Resize to model input size (64x64)
                original = cv2.resize(original, (64, 64))
                noisy = cv2.resize(noisy, (64, 64))
                
                original_images.append(original)
                noisy_images.append(noisy)
        
        print(f"Loaded {len(original_images)} images")
        
        if len(original_images) == 0:
            print("ERROR: Could not load any valid images!")
            return
        
        # Convert to numpy arrays
        X_train = np.array(noisy_images, dtype=np.float32)
        y_train = np.array(original_images, dtype=np.float32)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Target data shape: {y_train.shape}")
        
        # Training loop
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        
        batch_size = config['training']['batch_size']
        epochs = config['training']['epochs']
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        
        # Save trained model
        model_path = config['paths']['model_save']
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Plot training history
        plot_training_history(
            history,
            save_path=str(Path(config['paths']['outputs']) / "training_history.png")
        )
        
        # Save sample denoising results
        print("\n" + "="*60)
        print("Saving sample denoising results...")
        print("="*60)
        
        for i in range(min(5, len(X_train))):
            denoised = model.predict(X_train[i:i+1], verbose=0)[0]
            denoised = clip_image(denoised)
            
            plot_denoising_result(
                X_train[i],
                denoised,
                y_train[i],
                save_path=str(Path(config['paths']['outputs']) / f"result_{i}.png")
            )
            
            # Calculate metrics
            psnr = calculate_psnr(y_train[i], denoised)
            ssim = calculate_ssim(y_train[i], denoised)
            print(f"Sample {i}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Make sure preprocessing.py functions are available")
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()