import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


def create_unet(in_channels=3, out_channels=3, features=64):
    
    input_layer = keras.Input(shape=(64, 64, in_channels))
    
    conv1a = layers.Conv2D(features, 3, padding='same', activation='relu')(input_layer)
    conv1b = layers.Conv2D(features, 3, padding='same', activation='relu')(conv1a)
    pool1 = layers.MaxPooling2D(2)(conv1b)
    
    conv2a = layers.Conv2D(features * 2, 3, padding='same', activation='relu')(pool1)
    conv2b = layers.Conv2D(features * 2, 3, padding='same', activation='relu')(conv2a)
    pool2 = layers.MaxPooling2D(2)(conv2b)
    
    conv3a = layers.Conv2D(features * 4, 3, padding='same', activation='relu')(pool2)
    conv3b = layers.Conv2D(features * 4, 3, padding='same', activation='relu')(conv3a)
    pool3 = layers.MaxPooling2D(2)(conv3b)
    
    conv4a = layers.Conv2D(features * 8, 3, padding='same', activation='relu')(pool3)
    conv4b = layers.Conv2D(features * 8, 3, padding='same', activation='relu')(conv4a)
    pool4 = layers.MaxPooling2D(2)(conv4b)
    
    bottleneck = layers.Conv2D(features * 16, 3, padding='same', activation='relu')(pool4)
    bottleneck = layers.Conv2D(features * 8, 3, padding='same', activation='relu')(bottleneck)
    
    up4 = layers.Conv2DTranspose(features * 4, 2, strides=2, padding='same')(bottleneck)
    concat4 = layers.Concatenate()([up4, conv4b])
    dec4a = layers.Conv2D(features * 4, 3, padding='same', activation='relu')(concat4)
    dec4b = layers.Conv2D(features * 4, 3, padding='same', activation='relu')(dec4a)
    
    up3 = layers.Conv2DTranspose(features * 2, 2, strides=2, padding='same')(dec4b)
    concat3 = layers.Concatenate()([up3, conv3b])
    dec3a = layers.Conv2D(features * 2, 3, padding='same', activation='relu')(concat3)
    dec3b = layers.Conv2D(features * 2, 3, padding='same', activation='relu')(dec3a)
    
    up2 = layers.Conv2DTranspose(features, 2, strides=2, padding='same')(dec3b)
    concat2 = layers.Concatenate()([up2, conv2b])
    dec2a = layers.Conv2D(features, 3, padding='same', activation='relu')(concat2)
    dec2b = layers.Conv2D(features, 3, padding='same', activation='relu')(dec2a)
    
    up1 = layers.Conv2DTranspose(features, 2, strides=2, padding='same')(dec2b)
    concat1 = layers.Concatenate()([up1, conv1b])
    dec1a = layers.Conv2D(features, 3, padding='same', activation='relu')(concat1)
    dec1b = layers.Conv2D(features, 3, padding='same', activation='relu')(dec1a)
    
    output = layers.Conv2D(out_channels, 1, padding='same')(dec1b)
    
    model = Model(inputs=input_layer, outputs=output, name='UNet')
    return model


def create_autoencoder(in_channels=3, out_channels=3):
    
    input_layer = keras.Input(shape=(64, 64, in_channels))
    
    enc1 = layers.Conv2D(64, 3, padding='same', activation='relu')(input_layer)
    enc1 = layers.MaxPooling2D(2)(enc1)
    
    enc2 = layers.Conv2D(128, 3, padding='same', activation='relu')(enc1)
    enc2 = layers.MaxPooling2D(2)(enc2)
    
    enc3 = layers.Conv2D(256, 3, padding='same', activation='relu')(enc2)
    
    dec1 = layers.Conv2D(128, 3, padding='same', activation='relu')(enc3)
    dec1 = layers.UpSampling2D(2)(dec1)
    
    dec2 = layers.Conv2D(64, 3, padding='same', activation='relu')(dec1)
    dec2 = layers.UpSampling2D(2)(dec2)
    
    output = layers.Conv2D(out_channels, 3, padding='same')(dec2)
    
    model = Model(inputs=input_layer, outputs=output, name='DenoisingAutoencoder')
    return model


def count_parameters(model):
    return model.count_params()


def print_model_info(model, model_name="Model"):
    total_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"{model_name} Architecture Info")
    print(f"{'='*60}")
    model.summary()
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print(f"Using TensorFlow: {tf.__version__}")
    
    model = create_unet(in_channels=3, out_channels=3, features=64)
    print_model_info(model, "U-Net Denoiser")
    
    dummy_input = np.random.randn(2, 64, 64, 3).astype(np.float32)
    output = model.predict(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    autoencoder = create_autoencoder(in_channels=3, out_channels=3)
    print_model_info(autoencoder, "Denoising Autoencoder")
    output_ae = autoencoder.predict(dummy_input)
    print(f"Autoencoder output shape: {output_ae.shape}")
