import numpy as np
import cv2
import random

def salt_and_pepper(image, amount=0.05):
    # p=[tuz, biber, orijinal]
    p_orig = 1 - amount
    p_noise = amount / 2
    noise = np.random.choice([0, 1, 0.5], size=image.shape, p=[p_noise, p_noise, p_orig])
    
    noisy_img = image.copy()
    noisy_img[noise == 0] = 0
    noisy_img[noise == 1] = 1

    return np.clip(noisy_img, 0, 1)

def speckle_noise(image, sigma=0.1): # sigma parametresi eklendi
    row, col, ch = image.shape
    gauss = np.random.normal(0, sigma, (row, col, ch)) # 0.1 yerine sigma kullanıldı
    noisy_img = image + image * gauss
    return np.clip(noisy_img, 0, 1)

def stripe_noise(image, sigma=0.1):
    h, w, c = image.shape

    # Tüm kanallara aynı çizgiyi ekle (daha gerçekçi uydu hatası)
    noise_vector = np.random.normal(0, sigma, (h, 1, 1))

    # 2. Add noise to the image (NumPy broadcasting handles the width)
    noisy_img = image + noise_vector

    return np.clip(noisy_img, 0, 1)

def poisson_noise(image, peak=90.0): # peak parametresi eklendi
    scaled_image = image * peak # 10.0 yerine peak kullanıldı
    noisy = np.random.poisson(scaled_image)
    noisy_img = noisy / peak
    return np.clip(noisy_img, 0, 1)

def cosmic_ray(image):
    x1, y1 = np.random.randint(0, 64, 2)
    x2, y2 = x1 + np.random.randint(-5, 5), y1 + np.random.randint(-5, 5)
    cv2.line(image, (x1, y1), (x2, y2), (1.0, 1.0, 1.0), 1) # Parlak beyaz çizgi

    return image

def random_noise(image):
    rand_num = random.random()

    if rand_num < 0.2: # %20 ihtimalle buraya girer
        return image # Resme hiç dokunmadan geri gönder (Identity Mapping)
    
    noisy_img = image.copy()
    
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
        pk = random.uniform(50.0, 150.0)
        noisy_img = poisson_noise(noisy_img, peak=pk)
        
    if random.random() < 0.3:
        noisy_img = cosmic_ray(noisy_img)
        
    return noisy_img