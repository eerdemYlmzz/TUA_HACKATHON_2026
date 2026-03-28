import numpy as np
import cv2

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

def poisson_noise(image, peak=10.0): # peak parametresi eklendi
    scaled_image = image * peak # 10.0 yerine peak kullanıldı
    noisy = np.random.poisson(scaled_image)
    noisy_img = noisy / peak
    return np.clip(noisy_img, 0, 1)

def cosmic_ray(image):
    x1, y1 = np.random.randint(0, 64, 2)
    x2, y2 = x1 + np.random.randint(-5, 5), y1 + np.random.randint(-5, 5)
    cv2.line(image, (x1, y1), (x2, y2), (1.0, 1.0, 1.0), 1) # Parlak beyaz çizgi

    return image