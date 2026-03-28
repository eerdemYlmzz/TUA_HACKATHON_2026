import tensorflow as tf
import numpy as np
import yaml
import cv2
import os
from pathlib import Path

from src.models import create_unet, create_autoencoder, print_model_info
from src.preprocessing import select_images, load_and_add_noise, get_train_val_test_paths
from src.utils import (
    plot_denoising_result, plot_training_history,
    calculate_psnr, calculate_ssim, clip_image
)

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def paths_to_pixels_numpy(path_list, config):
    """Dosya yollarını okuyup gürültülü/temiz numpy dizilerine çevirir."""
    clean_list = []
    noisy_list = []
    noise_type = config['params'].get('noise_type', 'random')
    
    for p in path_list:
        clean, noisy = load_and_add_noise(p, noise_type=noise_type)
        if clean is not None:
            # Model girişi 64x64 olduğu için boyutlandırma garantisi
            clean = cv2.resize(clean, (64, 64))
            noisy = cv2.resize(noisy, (64, 64))
            clean_list.append(clean)
            noisy_list.append(noisy)
            
    return np.array(noisy_list, dtype=np.float32), np.array(clean_list, dtype=np.float32)

def main():
    # 1. Ayarları Yükle
    config = load_config()
    Path(config['paths']['outputs']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['models']).mkdir(parents=True, exist_ok=True)

    # 2. Veri Hazırlığı (Data Preparation)
    print("\n[1/4] Veri yolları hazırlanıyor...")
    all_paths = select_images(config['paths']['raw_data'], n_samples=200) # Her sınıftan 200 örnek
    
    # Yolları %70 Train, %15 Val, %15 Test olarak ayır
    train_p, val_p, test_p = get_train_val_test_paths(all_paths)
    
    print(f"Toplam: {len(all_paths)} | Train: {len(train_p)} | Val: {len(val_p)} | Test: {len(test_p)}")

    # Yolları gerçek veriye (numpy) dönüştür
    print("Veriler belleğe yükleniyor (Gürültü ekleniyor)...")
    X_train, y_train = paths_to_pixels_numpy(train_p, config)
    X_val, y_val = paths_to_pixels_numpy(val_p, config)
    X_test, y_test = paths_to_pixels_numpy(test_p, config)

    # 3. Model Kurulumu ve Eğitim
    print("\n[2/4] Model başlatılıyor...")
    model = create_unet(features=config['model']['unet']['features'])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    model.compile(optimizer=optimizer, loss=config['training']['loss_function'], metrics=['mae'])

    print("\n[3/4] Eğitim başlıyor...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val), # Doğrulama seti burada kullanılır
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        verbose=1
    )

    # Modeli kaydet
    model.save(config['paths']['model_save'])
    plot_training_history(history, save_path=os.path.join(config['paths']['outputs'], "loss_chart.png"))

    # 4. Bağımsız Test ve Değerlendirme
    print("\n[4/4] Model test ediliyor (Unseen Data)...")
    results = model.predict(X_test)
    results = np.clip(results, 0, 1)

    avg_psnr = 0
    avg_ssim = 0

    # Test sonuçlarını kaydet ve metrikleri hesapla
    for i in range(min(10, len(X_test))):
        psnr = calculate_psnr(y_test[i], results[i])
        ssim = calculate_ssim(y_test[i], results[i])
        avg_psnr += psnr
        avg_ssim += ssim
        
        plot_denoising_result(
            X_test[i], results[i], y_test[i],
            save_path=os.path.join(config['paths']['outputs'], f"test_result_{i}.png")
        )

    print("\n" + "="*30)
    print(f"TEST SONUÇLARI (Ortalama)")
    print(f"PSNR: {avg_psnr / min(10, len(X_test)):.2f} dB")
    print(f"SSIM: {avg_ssim / min(10, len(X_test)):.4f}")
    print("="*30)

if __name__ == "__main__":
    main()