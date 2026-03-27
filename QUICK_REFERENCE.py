import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import create_unet, create_autoencoder, print_model_info
from utils import (
    save_model, load_model, CombinedLoss,
    plot_denoising_result, plot_training_history, plot_batch_results,
    calculate_psnr, calculate_ssim, calculate_mse,
    normalize_image, denormalize_image, clip_image,
    tensor_to_numpy, numpy_to_tensor
)


model = create_unet(in_channels=3, out_channels=3, features=64)
print_model_info(model, "U-Net Denoiser")


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = CombinedLoss(weight_l1=0.5, weight_mse=0.5)
model.compile(optimizer=optimizer, loss=loss_fn)


epochs = 50
train_losses = []
val_losses = []

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1
)

train_losses = history.history['loss']
val_losses = history.history['val_loss']

plot_training_history(train_losses, val_losses, save_path="outputs/training_history.png")


noisy_normalized = normalize_image(noisy_image, max_value=1.0)
noisy_tensor = numpy_to_tensor(noisy_normalized[np.newaxis, ...])
denoised_tensor = model(noisy_tensor, training=False)
denoised = tensor_to_numpy(denoised_tensor[0])
denoised = clip_image(denoised, min_val=0.0, max_val=1.0)

if original_image is not None:
    psnr = calculate_psnr(original_image, denoised)
    ssim = calculate_ssim(original_image, denoised)
    print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")


plot_denoising_result(
    noisy=noisy_image,
    denoised=denoised,
    original=original_image,
    title="Satellite Image Denoising",
    save_path="outputs/denoising_result.png"
)


batch_noisy = numpy_to_tensor(batch_noisy)
batch_denoised = model(batch_noisy, training=False)

plot_batch_results(
    batch_noisy=tensor_to_numpy(batch_noisy),
    batch_denoised=tensor_to_numpy(batch_denoised),
    batch_original=batch_original,
    num_samples=4,
    save_path="outputs/batch_results.png"
)


model = create_unet(in_channels=3, out_channels=3, features=64)
metadata = load_model(model, "models/unet_denoiser_best.h5")
print(f"Loaded model from epoch {metadata['epoch']}")
print(f"Best validation loss was: {metadata['metrics']['val_loss']:.6f}")


psnr = calculate_psnr(original_image, denoised_image)
ssim = calculate_ssim(original_image, denoised_image)
mse = calculate_mse(original_image, denoised_image)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
print(f"MSE: {mse:.6f}")


import cv2

noisy = cv2.imread("noisy.jpg", cv2.IMREAD_COLOR).astype(np.float32) / 255.0
original = cv2.imread("original.jpg", cv2.IMREAD_COLOR).astype(np.float32) / 255.0

denoised_classical = cv2.fastNlMeansDenoisingColored(
    (noisy * 255).astype(np.uint8),
    None,
    h=10,
    templateWindowSize=7,
    searchWindowSize=21
).astype(np.float32) / 255.0

psnr_classical = calculate_psnr(original, denoised_classical)
psnr_unet = calculate_psnr(original, denoised_unet)

print(f"Classical Method PSNR: {psnr_classical:.2f} dB")
print(f"U-Net Method PSNR: {psnr_unet:.2f} dB")
print(f"Improvement: {psnr_unet - psnr_classical:.2f} dB")


image_np = np.random.rand(3, 64, 64).astype(np.float32)
image_tensor = numpy_to_tensor(image_np)
image_back = tensor_to_numpy(image_tensor)


image_normalized = normalize_image(image_raw, max_value=1.0)
image_original = denormalize_image(image_normalized, original_data_range=(0, 255))


cv2.imwrite("outputs/denoised.jpg", (denoised * 255).astype(np.uint8))
np.save("outputs/denoised.npy", denoised)

save_model(
    model,
    "models/final_model.h5",
    epoch=final_epoch,
    metrics={'psnr': psnr, 'ssim': ssim, 'mse': mse}
)


import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
epochs = config['training']['epochs']
model_save_path = config['paths']['model_save']

Path(config['paths']['outputs']).mkdir(parents=True, exist_ok=True)
Path(config['paths']['models']).mkdir(parents=True, exist_ok=True)
