import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from src.models import create_unet
from src.utils import normalize_image, clip_image, calculate_psnr, calculate_ssim, calculate_mse

# Page config
st.set_page_config(
    page_title="Satellite Image Denoiser",
    page_icon="🛰", #değiştiremedim. png de eklenebiliyor.
    layout="wide"
)

# Title
st.title(" Satellite Image Denoiser")
st.markdown("Clean satellite imagery using deep learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Settings")
    noise_type = st.selectbox(
        "Noise Type:",
        ["salt_and_pepper", "speckle", "stripe", "poisson", "cosmic_ray", "random"]
    )
    st.markdown("---")
    st.info("Model: U-Net\n\nParameters: 18M+\n\nInput: 64×64×3")

@st.cache_resource
def load_model():
    """Load trained model"""
    model_path = Path(__file__).parent / "models" / "denoiser_v1.h5"
    
    if not Path(model_path).exists():
        return None
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Hata: {type(e).__name__}: {e}")
        return None

def add_noise_to_image(image, noise_type="salt_and_pepper"):
    """Add noise to image"""
    try:
        from src.noises import salt_and_pepper, speckle_noise, stripe_noise, poisson_noise, cosmic_ray, random_noise
        
        funcs = {
            "salt_and_pepper": salt_and_pepper,
            "speckle": speckle_noise,
            "stripe": stripe_noise,
            "poisson": poisson_noise,
            "cosmic_ray": cosmic_ray,
            "random": random_noise
        }
        
        return funcs.get(noise_type, salt_and_pepper)(image)
    except:
        return image

def process_image(model, image_input, noise_type="salt_and_pepper"):
    """Process image through model"""
    try:
        # Normalize
        if image_input.max() > 1.0:
            image_input = image_input.astype(np.float32) / 255.0
        
        # Ensure RGB
        if len(image_input.shape) == 2:
            image_input = np.stack([image_input] * 3, axis=-1)
        elif image_input.shape[2] == 4:
            image_input = image_input[:, :, :3]

        image_64 = cv2.resize(image_input, (64, 64))

        # Add noise
        noisy_image = add_noise_to_image(image_64, noise_type)
        
        # Resize
        original_shape = image_input.shape[:2]
        
        # Denoise noisy input (not the clean image)
        denoised_64 = model.predict(np.expand_dims(noisy_image, 0), verbose=0)[0]
        denoised_64 = clip_image(denoised_64)
        
        # Resize back
        denoised = cv2.resize(denoised_64, (original_shape[1], original_shape[0]))
        
        # Metrics
        noisy_mse = calculate_mse(image_64, noisy_image)
        denoised_mse = calculate_mse(image_64, denoised_64)
        noisy_psnr = calculate_psnr(image_64, noisy_image)
        denoised_psnr = calculate_psnr(image_64, denoised_64)
        noisy_ssim = calculate_ssim(image_64, noisy_image)
        denoised_ssim = calculate_ssim(image_64, denoised_64)

        # Keep the same display size as original for easier visual comparison
        noisy = cv2.resize(noisy_image, (original_shape[1], original_shape[0]))
        
        return {
            'original': image_input,
            'noisy': noisy,
            'denoised': denoised,
            'metrics': {
                'noisy_mse': noisy_mse,
                'mse': denoised_mse,
                'noisy_psnr': noisy_psnr,
                'psnr': denoised_psnr,
                'noisy_ssim': noisy_ssim,
                'ssim': denoised_ssim,
            }
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Load model
model = load_model()

if model is not None:
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Upload", "Samples", "Info"])
    
    with tab1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            with st.spinner("Processing..."):
                result = process_image(model, image_array, noise_type)
            
            if result:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original")
                    st.image(result['original'], use_column_width=True)
                
                with col2:
                    st.subheader("Noisy")
                    st.image(result['noisy'], use_column_width=True)
                
                with col3:
                    st.subheader("Denoised")
                    st.image(result['denoised'], use_column_width=True)
                
                st.markdown("---")
                st.subheader("Metrics")
                
                metrics = result['metrics']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "MSE (Denoised)",
                        f"{metrics['mse']:.4f}",
                        delta=f"{(metrics['noisy_mse'] - metrics['mse']):.4f}"
                    )
                
                with col2:
                    st.metric(
                        "PSNR (Denoised)",
                        f"{metrics['psnr']:.2f} dB",
                        delta=f"{(metrics['psnr'] - metrics['noisy_psnr']):.2f} dB"
                    )
                
                with col3:
                    st.metric(
                        "SSIM (Denoised)",
                        f"{metrics['ssim']:.4f}",
                        delta=f"{(metrics['ssim'] - metrics['noisy_ssim']):.4f}"
                    )

                st.markdown("#### Noisy vs Denoised Comparison")
                st.table([
                    {
                        "Metric": "MSE (lower is better)",
                        "Noisy": f"{metrics['noisy_mse']:.4f}",
                        "Denoised": f"{metrics['mse']:.4f}",
                        "Improvement": f"{(metrics['noisy_mse'] - metrics['mse']):.4f}"
                    },
                    {
                        "Metric": "PSNR dB (higher is better)",
                        "Noisy": f"{metrics['noisy_psnr']:.2f}",
                        "Denoised": f"{metrics['psnr']:.2f}",
                        "Improvement": f"+{(metrics['psnr'] - metrics['noisy_psnr']):.2f}"
                    },
                    {
                        "Metric": "SSIM (higher is better)",
                        "Noisy": f"{metrics['noisy_ssim']:.4f}",
                        "Denoised": f"{metrics['ssim']:.4f}",
                        "Improvement": f"+{(metrics['ssim'] - metrics['noisy_ssim']):.4f}"
                    }
                ])
    
    with tab2:
        st.subheader("Sample Results")
        
        from glob import glob
        results = sorted(glob("outputs/result_*.png"))
        
        if results:
            st.write(f"Found {len(results)} samples")
            for result_path in results:
                st.image(result_path, use_column_width=True)
        else:
            st.info("No samples found")
    
    with tab3:
        st.subheader("Model Info")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Architecture:** U-Net")
            st.write("**Parameters:** 18M+")
            st.write("**Input:** 64×64×3")
        
        with col2:
            st.write("**Epochs:** 50")
            st.write("**Batch Size:** 32")
            st.write("**Loss:** MSE")
        
        if Path("outputs/training_history.png").exists():
            st.image("outputs/training_history.png", use_column_width=True)

else:
    st.error("Model not found! Run: python main.py")
