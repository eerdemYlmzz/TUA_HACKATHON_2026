import tensorflow as tf
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import create_unet, create_autoencoder, print_model_info
from utils import (
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
        from utils import CombinedLoss
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
    
    print("\n" + "="*60)
    print("NOTE: To start training, you need to:")
    print("1. Create preprocessing.py with data loading functions")
    print("2. Create noises.py with noise injection functions")
    print("3. Implement the training loop below")
    print("="*60)
    
    Path(config['paths']['outputs']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['logs']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['models']).mkdir(parents=True, exist_ok=True)
    
    print("\nSetup complete! Model is ready for training.")
    print(f"Model saved to: {config['paths']['model_save']}")
    print(f"Outputs will be saved to: {config['paths']['outputs']}")


if __name__ == "__main__":
    main()
