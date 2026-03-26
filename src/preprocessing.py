import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.noises import salt_and_pepper, speckle_noise, stripe_noise, poisson_noise

def select_images(data_path, n_samples=500):
    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    images_to_use = []

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        all_imgs = os.listdir(cls_path)
        # Klasördeki resim sayısı n_samples'dan azsa hepsini al
        actual_samples = min(len(all_imgs), n_samples)
        selected = random.sample(all_imgs, actual_samples) 
        for img in selected:
            images_to_use.append(os.path.join(cls_path, img))
            
    return images_to_use

def load_and_add_noise(image_path, noise_type="salt_and_pepper"):
    # 1. Resmi yükle ve RGB'ye çevir
    img = cv2.imread(image_path)

    if img is None: return None, None # Hatalı dosya kontrolü

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Görsel veriyi ölçeklendir
    img = normalize_image(img)

    # 3. Görsele gürültü çeşidine göre gürültü ekle
    noise_map = {
        "salt_and_pepper": salt_and_pepper,
        "speckle": speckle_noise,
        "stripe": stripe_noise,
        "poisson": poisson_noise
    }

    # noise_type geçerli değilse salt and pepper ekle
    noise_method = noise_map.get(noise_type, salt_and_pepper)
    noisy_img = noise_method(img)
    
    return img, noisy_img

def normalize_image(image):
    return image.astype(np.float32) / 255.0