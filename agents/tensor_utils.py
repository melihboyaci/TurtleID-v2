"""
tensor_utils.py — Tensör Dönüşüm Yardımcı Modülü
===================================================

RGB numpy görsellerini derin öğrenme modeline (ResNet50 tabanlı
embedding ağı) hazır tensörlere dönüştüren fonksiyonları içerir.

Bu modül, PreprocessingWorker ve RecognitionWorker tarafından
ortaklaşa kullanılır. Worker modülleri arasında doğrudan import
bağımlılığı oluşmaması için ayrı bir utility olarak tasarlanmıştır.

Kullanım:
    from agents.tensor_utils import to_tensor

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Bu modül yalnızca tensör hazırlığı yapar.
# DRY  : Hem PreprocessingWorker (query) hem RecognitionWorker
#        (veritabanı) aynı dönüşüm mantığını bu tek fonksiyondan alır.
# DIP  : Worker'lar birbirinin modülüne değil, bu bağımsız yardımcı
#        modüle bağımlıdır → gevşek bağlılık (loose coupling) sağlanır.
# ─────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

from config import TARGET_SIZE


def to_tensor(img_rgb: np.ndarray) -> np.ndarray:
    """
    RGB numpy görselini ResNet50 embedding modeline hazır tensöre dönüştürür.

    Dönüşüm Adımları:
        1. Görsel boyutunu TARGET_SIZE'a (varsayılan 224×224) ölçekler.
        2. (H, W, 3) → (1, H, W, 3) şekline batch boyutu ekler.
        3. float32'ye çevirir.
        4. ResNet50'ye özgü preprocess_input (kanal bazlı ortalama çıkarma)
           uygular.

    Args:
        img_rgb: (H, W, 3) şeklinde, 0-255 aralığında uint8 RGB numpy dizisi.

    Returns:
        (1, 224, 224, 3) şeklinde float32 tensör; ResNet50 ön-işlemesi
        uygulanmış halde.
    """
    if img_rgb.shape[:2] != TARGET_SIZE:
        img_rgb = cv2.resize(img_rgb, TARGET_SIZE)
    tensor = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    return preprocess_input(tensor)
