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
        ["salt_and_pepper", "speckle", "stripe", "poisson"]
    )
    st.markdown("---")
    st.info("Model: U-Net\n\nParameters: 18M+\n\nInput: 64×64×3")

@st.cache_resource
def load_model():
    """Load trained model"""
    model_path = "./models/denoiser_v1.h5"
    
    if not Path(model_path).exists():
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        return None

def add_noise_to_image(image, noise_type="salt_and_pepper"):
    """Add noise to image"""
    try:
        from src.noises import salt_and_pepper, speckle_noise, stripe_noise, poisson_noise
        
        funcs = {
            "salt_and_pepper": salt_and_pepper,
            "speckle": speckle_noise,
            "stripe": stripe_noise,
            "poisson": poisson_noise
        }
        
        return funcs[noise_type](image)
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
        
        # Add noise
        noisy_image = add_noise_to_image(image_input, noise_type)
        
        # Resize
        original_shape = image_input.shape[:2]
        image_64 = cv2.resize(noisy_image, (64, 64))
        
        # Denoise
        denoised_64 = model.predict(np.expand_dims(image_64, 0), verbose=0)[0]
        denoised_64 = clip_image(denoised_64)
        
        # Resize back
        denoised = cv2.resize(denoised_64, (original_shape[1], original_shape[0]))
        
        # Metrics
        mse = calculate_mse(image_64, image_64)
        noisy_mse = calculate_mse(image_64, image_64)
        denoised_mse = calculate_mse(image_64, denoised_64)
        psnr = calculate_psnr(image_64, denoised_64)
        ssim = calculate_ssim(image_64, denoised_64)
        
        return {
            'original': image_input,
            'noisy': noisy_image,
            'denoised': denoised,
            'metrics': {
                'mse': denoised_mse,
                'psnr': psnr,
                'ssim': ssim,
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
                    st.metric("MSE", f"{metrics['mse']:.4f}")
                
                with col2:
                    st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                
                with col3:
                    st.metric("SSIM", f"{metrics['ssim']:.4f}")
    
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
