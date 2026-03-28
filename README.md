# 🛰️ TUA Hackathon 2026 — Uydu Görüntüsü Gürültü Giderici (Denoiser)

Bu proje, Sentinel-2 uydu görüntülerindeki radyasyon hasarı, atmosferik bozulmalar ve iletim hatalarını temizlemek amacıyla **U-Net mimarisi** kullanılarak geliştirilmiş bir derin öğrenme çözümüdür. TUA Hackathon 2026 kapsamında, uzaydan gelen optik verilerin kalitesini artırmak için tasarlanmıştır.

---

## 🎯 Misyonumuz

> *"Uzaydan gelen veriler, atmosferik olaylar veya radyasyon nedeniyle bozulabiliyor. Biz, Sentinel-2 uydusundan gelen optik verileri, sanki en pahalı donanımsal filtrelerden geçmiş gibi yapay zeka ile temizleyerek veri kaybının önüne geçiyoruz."*

---

## 📁 Proje Yapısı
```plaintext
TUA_HACKATHON_2026/
├── src/
│   ├── models.py          # U-Net (18M+ Parametre) ve Autoencoder mimarileri
│   ├── noises.py          # Sentetik gürültü üretim fonksiyonları
│   ├── preprocessing.py   # Veri yükleme, normalizasyon ve veri seti bölme
│   └── utils.py           # PSNR, SSIM metrikleri ve görselleştirme araçları
├── data/
│   ├── raw/               # Orijinal temiz EuroSAT görüntüleri
│   └── processed/         # İşlenmiş veriler
├── models/
│   └── denoiser_v1.h5     # Eğitilmiş TensorFlow model dosyası
├── outputs/               # Kaydedilen test sonuçları ve eğitim grafikleri
├── app.py                 # Streamlit web arayüzü
├── main.py                # Eğitim ve test ana çalıştırma scripti
├── config.yaml            # Model ve eğitim hiperparametreleri
└── requirements.txt       # Gerekli kütüphaneler
```

---

## 🛠️ Teknik Bileşenler

### 1. Model Mimarisi (`models.py`)

- **U-Net:** 4 katmanlı derin encoder-decoder yapısı. Atlamalı bağlantılar (skip connections) sayesinde düşük seviyeli detaylar korunur. Yaklaşık **18 milyon** eğitilebilir parametreye sahiptir.
- **Denoising Autoencoder:** Daha hızlı çıkarım (inference) için hafif bir alternatif model.

### 2. Gürültü Simülasyonu (`noises.py`)

Uzay operasyonlarında karşılaşılan gerçekçi hata türleri simüle edilmektedir:

| Gürültü Türü | Açıklama |
|---|---|
| **Salt & Pepper** | İletim hataları |
| **Speckle** | Koherent görüntüleme sistemleri paraziti |
| **Stripe Noise** | Sensör hatalarından kaynaklanan çizgilenmeler |
| **Poisson** | Düşük ışık / foton sayımı gürültüsü |
| **Cosmic Ray** | Radyasyon kaynaklı parlak beyaz çizgiler |

### 3. Veri İşleme (`preprocessing.py`)

- Görüntüleri otomatik olarak **64×64** boyutuna getirir.
- Eğitim sırasında verileri rastgele gürültü türleriyle (*random noise augmentation*) zenginleştirir.
- Veri setini otomatik olarak **%70 Eğitim / %15 Doğrulama / %15 Test** olarak ayırır.

---

## 🌐 Web Arayüzü (Streamlit)

Kullanıcı dostu arayüz sayesinde modelinizi kolayca test edebilirsiniz.

**Çalıştırma:**
```bash
streamlit run app.py
```

**Özellikler:**

- 📤 **Görsel Yükleme** — Kendi uydu görüntünüzü yükleyip temizleyebilirsiniz.
- 🧪 **Gürültü Seçimi** — Farklı gürültü tiplerinde modelin performansını canlı görün.
- 📊 **Metrik Takibi** — PSNR (dB), SSIM ve MSE değerlerini anlık hesaplar.
- 📉 **Eğitim Geçmişi** — Modelin öğrenme eğrisini ve örnek sonuçları inceleyin.

---

## 🚀 Hızlı Başlangıç

### Adım 1 — Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 2 — Eğitimi Başlatın

`config.yaml` dosyasından ayarlarınızı yapın ve modeli eğitin:
```bash
python main.py
```

### Adım 3 — Sonuçları İzleyin

Eğitim sonrası `outputs/` klasöründe oluşan `loss_chart.png` ve `test_result_x.png` dosyalarını inceleyerek modelin başarısını (PSNR/SSIM) kontrol edin.

---

## 📊 Beklenen Performans

| Parametre | Değer |
|---|---|
| **Giriş Boyutu** | 64×64×3 (RGB) |
| **Hedef PSNR** | 25 – 35 dB |
| **Hedef SSIM** | 0.85 – 0.95 |
| **Platform** | TensorFlow 2.x |

---

<div align="center">
  🚀 <strong>TUA Astro Hackathon — Adana Projesi</strong> 🚀
</div>
