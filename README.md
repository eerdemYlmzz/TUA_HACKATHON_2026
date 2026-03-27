# TUA Hackathon 2026 - Satellite Image Denoising

## 🛰️ Project Overview

A deep learning project for denoising **Sentinel-2 satellite imagery** using a **U-Net encoder-decoder architecture**. This project is designed for the TUA Hackathon 2026 with a focus on cleaning corrupted satellite data that resulted from radiation damage or transmission errors during space operations.

### 🎯 Mission Statement

**"Uzaydan gelen veriler, atmosferik olaylar veya radyasyon nedeniyle bozulabiliyor. Biz, Sentinel-2 uydusundan gelen optik verileri, sanki en pahalı donanımsal filtrelerden geçmiş gibi yapay zeka ile temizleyerek veri kaybının önüne geçiyoruz."**

*"Data coming from space can be corrupted by atmospheric events or radiation. We clean optical data from the Sentinel-2 satellite using artificial intelligence, as if it passed through the most expensive hardware filters, preventing data loss."*

---

## 📁 Project Structure

```
TUA_HACKATHON_2026/
├── data/
│   ├── raw/                    # Original clean EuroSAT images
│   └── processed/              # Images with synthetic noise
├── models/
│   └── unet_denoiser.pt       # Trained model checkpoint
├── notebooks/                  # Jupyter notebooks for exploration
├── outputs/
│   ├── logs/                   # Training logs
│   ├── training_history.png    # Loss curves
│   └── denoised_images/        # Output denoised images
├── src/
│   ├── models.py              # U-Net and Autoencoder architectures ✅
│   ├── noises.py              # Noise injection functions (to be created)
│   ├── preprocessing.py       # Data loading and augmentation (to be created)
│   └── utils.py               # Metrics, visualization, utilities ✅
├── config.yaml                # Configuration file ✅
├── main.py                    # Training entry point ✅
├── requirements.txt           # Dependencies ✅
├── .gitignore                 # Git ignore rules ✅
└── README.md                  # This file
```

---

## 🔧 Core Components (COMPLETED)

### ✅ models.py - Neural Network Architectures

**U-Net Architecture** (Primary Model)
- Encoder path: 4 layers of downsampling
- Bottleneck: Compressed feature representation
- Decoder path: 4 layers of upsampling with skip connections
- Total parameters: ~1.9M

**Denoising Autoencoder** (Lighter Alternative)
- Simpler 3-layer encoder-decoder
- Faster training/inference

### ✅ utils.py - Complete Utilities

**Metrics**: PSNR, SSIM, MSE calculations
**Visualization**: Training plots, batch results, before/after comparisons
**Loss Functions**: MSELoss, L1Loss, CombinedLoss
**Model I/O**: Save/load checkpoints
**Tensor Utils**: NumPy-PyTorch conversions, normalization, clipping

### ✅ config.yaml - Full Configuration

Comprehensive settings for paths, noise parameters, training, evaluation, logging

### ✅ main.py - Training Template

Entry point with model initialization and training skeleton

---

## 🎓 To Be Implemented

### preprocessing.py (Your Friend - Team)
- Image loading and augmentation
- DataLoader creation
- Data normalization

### noises.py (Your Friend - Team)
- Salt & pepper noise
- Gaussian noise  
- Speckle noise
- Poisson noise
- Stripe noise
- Combined noise injection

---

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download EuroSAT dataset to `data/raw/`

3. Once preprocessing and noises modules are ready:
```bash
python main.py
```

---

## 📊 Expected Performance

- PSNR: 25-35 dB
- SSIM: 0.85-0.95
- Training time: 30-60 min (GPU)

---

## Our TUA Astro Hackathon Adana Project
