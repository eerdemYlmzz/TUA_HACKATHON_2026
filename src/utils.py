import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def calculate_psnr(original: np.ndarray, denoised: np.ndarray, max_pixel_value: float = 1.0) -> float:
    
    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)
    
    mse = np.mean((original - denoised) ** 2)
    
    if mse == 0:
        return 100.0
    
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return float(psnr)


def calculate_ssim(original: np.ndarray, denoised: np.ndarray, 
                  data_range: float = 1.0, window_size: int = 11) -> float:
    
    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    kernel_size = window_size
    sigma = 1.5
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    window = kernel @ kernel.T
    
    if len(original.shape) == 3:
        ssim_values = []
        for c in range(original.shape[2]):
            orig_c = original[:, :, c]
            denoise_c = denoised[:, :, c]
            ssim_c = _calculate_ssim_single_channel(orig_c, denoise_c, window, C1, C2)
            ssim_values.append(ssim_c)
        return float(np.mean(ssim_values))
    else:
        return float(_calculate_ssim_single_channel(original, denoised, window, C1, C2))


def _calculate_ssim_single_channel(original: np.ndarray, denoised: np.ndarray, 
                                   window: np.ndarray, C1: float, C2: float) -> float:
    
    kernel_size = window.shape[0]
    pad = kernel_size // 2
    
    orig_padded = np.pad(original, pad, mode='symmetric')
    denoise_padded = np.pad(denoised, pad, mode='symmetric')
    
    h, w = original.shape
    ssim_map = np.zeros((h, w))
    
    for i in range(h):
        for j in range(w):
            orig_window = orig_padded[i:i+kernel_size, j:j+kernel_size]
            denoise_window = denoise_padded[i:i+kernel_size, j:j+kernel_size]
            
            mu1 = np.sum(orig_window * window)
            mu2 = np.sum(denoise_window * window)
            
            sigma1_sq = np.sum((orig_window - mu1) ** 2 * window)
            sigma2_sq = np.sum((denoise_window - mu2) ** 2 * window)
            sigma12 = np.sum((orig_window - mu1) * (denoise_window - mu2) * window)
            
            numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
            
            ssim_map[i, j] = numerator / denominator if denominator != 0 else 0
    
    return float(np.mean(ssim_map))


def calculate_mse(original: np.ndarray, denoised: np.ndarray) -> float:
    
    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)
    mse = np.mean((original - denoised) ** 2)
    return float(mse)


def plot_denoising_result(noisy: np.ndarray, denoised: np.ndarray, original: np.ndarray = None,
                         title: str = "Denoising Result", figsize: Tuple[int, int] = (15, 5),
                         save_path: str = None) -> None:
    
    num_images = 3 if original is not None else 2
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    noisy_display = _prepare_for_display(noisy)
    denoised_display = _prepare_for_display(denoised)
    
    axes[0].imshow(noisy_display)
    axes[0].set_title("Noisy Input", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_display)
    axes[1].set_title("Denoised Output", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    if original is not None:
        original_display = _prepare_for_display(original)
        axes[2].imshow(original_display)
        axes[2].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        psnr = calculate_psnr(original, denoised)
        ssim = calculate_ssim(original, denoised)
        mse = calculate_mse(original, denoised)
        
        fig.suptitle(f"{title}\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | MSE: {mse:.6f}",
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_training_history(train_losses: List[float], val_losses: List[float] = None,
                         save_path: str = None, figsize: Tuple[int, int] = (10, 6)) -> None:
    
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    
    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.show()


def plot_batch_results(batch_noisy: np.ndarray, batch_denoised: np.ndarray,
                      batch_original: np.ndarray = None, num_samples: int = 4,
                      save_path: str = None) -> None:
    
    num_samples = min(num_samples, batch_noisy.shape[0])
    num_cols = 3 if batch_original is not None else 2
    
    fig = plt.figure(figsize=(5 * num_cols, 4 * num_samples))
    gs = gridspec.GridSpec(num_samples, num_cols, figure=fig)
    
    for i in range(num_samples):
        ax = fig.add_subplot(gs[i, 0])
        noisy_img = _prepare_for_display(batch_noisy[i])
        ax.imshow(noisy_img)
        ax.set_title(f"Sample {i+1} - Noisy", fontweight='bold')
        ax.axis('off')
        
        ax = fig.add_subplot(gs[i, 1])
        denoised_img = _prepare_for_display(batch_denoised[i])
        ax.imshow(denoised_img)
        ax.set_title(f"Sample {i+1} - Denoised", fontweight='bold')
        ax.axis('off')
        
        if batch_original is not None:
            ax = fig.add_subplot(gs[i, 2])
            original_img = _prepare_for_display(batch_original[i])
            ax.imshow(original_img)
            ax.set_title(f"Sample {i+1} - Original", fontweight='bold')
            ax.axis('off')
    
    fig.suptitle("Batch Denoising Results", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Batch results saved to: {save_path}")
    
    plt.show()


def _prepare_for_display(image: np.ndarray) -> np.ndarray:
    
    image = np.asarray(image)
    
    if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
        image = np.transpose(image, (1, 2, 0))
    
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    
    image = np.clip(image, 0, 1)
    
    if image.ndim == 2:
        return image
    else:
        return image


def save_model(model, save_path: str, epoch: int = None, metrics: dict = None) -> None:
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.save(save_path)
    print(f"Model saved to: {save_path}")
    
    if metrics is not None:
        import json
        metrics_path = Path(save_path).with_suffix('.json')
        metadata = {'epoch': epoch, 'metrics': metrics}
        with open(metrics_path, 'w') as f:
            json.dump(metadata, f)


def load_model(load_path: str, device: str = 'cpu'):
    
    model = tf.keras.models.load_model(load_path)
    
    metadata = {'model_type': 'TensorFlow Model', 'epoch': 0, 'metrics': {}}
    
    metrics_path = Path(load_path).with_suffix('.json')
    if metrics_path.exists():
        import json
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            metadata.update(data)
    
    print(f"Model loaded from: {load_path}")
    print(f"Model Type: {metadata['model_type']}")
    if metadata['epoch'] > 0:
        print(f"Last trained epoch: {metadata['epoch']}")
    
    return metadata


def normalize_image(image: np.ndarray, max_value: float = 1.0) -> np.ndarray:
    
    image = image.astype(np.float32)
    img_min = image.min()
    img_max = image.max()
    
    if img_max == img_min:
        return np.zeros_like(image)
    
    normalized = (image - img_min) / (img_max - img_min) * max_value
    return normalized


def denormalize_image(image: np.ndarray, original_data_range: Tuple[float, float]) -> np.ndarray:
    
    original_min, original_max = original_data_range
    denormalized = image * (original_max - original_min) + original_min
    return denormalized


def clip_image(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    
    return np.clip(image, min_val, max_val)


def tensor_to_numpy(tensor) -> np.ndarray:
    
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    return np.asarray(tensor)


def numpy_to_tensor(array: np.ndarray, dtype=tf.float32):
    
    return tf.convert_to_tensor(array, dtype=dtype)


class MSELoss(tf.keras.losses.Loss):
    
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))


class L1Loss(tf.keras.losses.Loss):
    
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))


class CombinedLoss(tf.keras.losses.Loss):
    
    def __init__(self, weight_l1=0.5, weight_mse=0.5):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_mse = weight_mse
    
    def call(self, y_true, y_pred):
        loss_l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
        loss_mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return self.weight_l1 * loss_l1 + self.weight_mse * loss_mse


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print("\nAvailable functions:")
    print("- Metrics: calculate_psnr, calculate_ssim, calculate_mse")
    print("- Visualization: plot_denoising_result, plot_training_history, plot_batch_results")
    print("- Model I/O: save_model, load_model")
    print("- Losses: MSELoss, L1Loss, CombinedLoss")
    print("- Batch utilities: tensor_to_numpy, numpy_to_tensor, normalize_image, etc.")
