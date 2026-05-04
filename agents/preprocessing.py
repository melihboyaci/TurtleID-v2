import os
import cv2
import numpy as np
from typing import Tuple
from agents import BaseWorker


class PreprocessingWorker(BaseWorker):
    """
    Kafa görselini normalize eder.
    Blackboard'dan head_crop okur, processed_image yazar.
    OCP: target_size parametrik — farklı model için değiştirmek gerekmez.
    """

    TARGET_SIZE: Tuple[int, int] = (224, 224)
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: tuple = (8, 8)
    SHARPEN_STRENGTH: float = 0.3

    def execute(self) -> bool:
        if self.bb.head_crop is None:
            self.bb.fail(self.name, "Kırpılmış kafa görseli yok.")
            return False

        processed = self._process(self.bb.head_crop)
        if processed is None:
            self.bb.fail(self.name, "Görsel işlenemedi.")
            return False

        self.bb.processed_image = processed

        os.makedirs("logs", exist_ok=True)
        cv2.imwrite(
            "logs/debug_enhanced_head.jpg",
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        )

        # DB görsellerini de işle
        self.bb.db_embeddings = []  # Recognition'da doldurulacak
        self.log(f"Görsel {self.TARGET_SIZE} boyutuna getirildi ve CLAHE uygulandı.")
        return True

    def enhance_scales(self, image: np.ndarray) -> np.ndarray:
        """
        Su altı kaplumbağa fotoğraflarındaki düşük kontrastı
        giderir ve pul sınırlarını belirginleştirir.

        Pipeline:
        1. RGB → LAB dönüşümü
        2. L kanalına CLAHE (kontrast iyileştirme)
        3. LAB → RGB geri dönüşüm
        4. Hafif Unsharp Masking (pul netleştirme)

        Not: clipLimit=2.0 bilinçli seçim — agresif CLAHE
        ResNet50'nin ImageNet dağılımından uzaklaştırır.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT,
            tileGridSize=self.CLAHE_TILE_SIZE
        )
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        if self.SHARPEN_STRENGTH > 0:
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(
                enhanced,
                1 + self.SHARPEN_STRENGTH,
                blurred,
                -self.SHARPEN_STRENGTH,
                0
            )

        return enhanced

    def _process(self, image: np.ndarray) -> np.ndarray:
        enhanced = self.enhance_scales(image)
        resized  = cv2.resize(enhanced, self.TARGET_SIZE)
        return resized
