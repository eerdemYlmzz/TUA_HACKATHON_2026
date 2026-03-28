import os
import random
import cv2
import numpy as np
from src.noises import salt_and_pepper, speckle_noise, stripe_noise, poisson_noise, cosmic_ray
from sklearn.model_selection import train_test_split

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

def load_and_add_noise(image_path, noise_type="random"):
    img = cv2.imread(image_path)
    if img is None: return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = normalize_image(img)
    
    # 1. Eğer noise_type "none" ise görseli hiç değiştirmeden döndür
    if noise_type == "none": return img, img

    # 2. Eğer noise_type "random" ise kombinasyon yapalım
    if noise_type == "random":
        rand_num = random.random()

        if rand_num < 0.2: # %20 ihtimalle buraya girer
            return img, img # Resme hiç dokunmadan geri gönder (Identity Mapping)
        
        noisy_img = img.copy()
        
        # Her gürültü türü kendi içinde bağımsız şansa sahip
        # (Buradaki oranları yüksek tutmak kombinasyon olasılığını artırır)

        if random.random() < 0.5:
            # Salt & Pepper: %1 ile %10 arasında rastgele yoğunluk
            amt = random.uniform(0.01, 0.10)
            noisy_img = salt_and_pepper(noisy_img, amount=amt)
            
        if random.random() < 0.4:
            # Speckle: 0.05 ile 0.2 arasında rastgele sigma
            sig = random.uniform(0.05, 0.2)
            noisy_img = speckle_noise(noisy_img, sigma=sig)
            
        if random.random() < 0.3:
            # Stripe: 0.05 ile 0.15 arasında rastgele çizgi şiddeti
            sig_s = random.uniform(0.05, 0.15)
            noisy_img = stripe_noise(noisy_img, sigma=sig_s)
            
        if random.random() < 0.4:
            # Poisson: Peak değerini 5 ile 20 arasında değiştirerek ışık gürültüsü ekle
            pk = random.uniform(5.0, 20.0)
            noisy_img = poisson_noise(noisy_img, peak=pk)
            
        if random.random() < 0.3:
            noisy_img = cosmic_ray(noisy_img)
            
        return img, noisy_img

    # 3. Belirli bir gürültü tipi istendiyse eski mantık devam eder
    noise_map = {
        "salt_and_pepper": salt_and_pepper,
        "speckle": speckle_noise,
        "stripe": stripe_noise,
        "poisson": poisson_noise,
        "cosmic_ray": cosmic_ray
    }
    
    noise_method = noise_map.get(noise_type, salt_and_pepper)
    noisy_img = noise_method(img)
    
    return img, noisy_img

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def get_train_val_test_paths(all_image_paths, train_size=0.7, val_size=0.15, test_size=0.15):
    # Önce Train ve Geriye Kalanlar (Val + Test) olarak ayır
    train_paths, remaining_paths = train_test_split(
        all_image_paths, train_size=train_size, random_state=42, shuffle=True
    )
    
    # Kalanları da Kendi İçinde Val ve Test olarak ikiye böl
    # (0.15 / 0.30 = 0.5 olduğu için test_size=0.5 diyoruz)
    val_paths, test_paths = train_test_split(
        remaining_paths, test_size=0.5, random_state=42, shuffle=True
    )
    
    return train_paths, val_paths, test_paths