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
    CROP_RATIO: float = 0.8

    def execute(self) -> bool:
        if self.bb.head_crop is None:
            self.bb.fail(self.name, "Kırpılmış kafa görseli yok.")
            return False

        processed = self._process(self.bb.head_crop)
        if processed is None:
            self.bb.fail(self.name, "Görsel işlenemedi.")
            return False

        self.bb.processed_image = processed

        # DB görsellerini de işle
        self.bb.db_embeddings = []  # Recognition'da doldurulacak
        self.log(f"Görsel {self.TARGET_SIZE} boyutuna getirildi.")
        return True

    def _process(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        nw, nh = int(w * self.CROP_RATIO), int(h * self.CROP_RATIO)
        left, top = (w - nw) // 2, (h - nh) // 2
        cropped = image[top:top+nh, left:left+nw]
        return cv2.resize(cropped, self.TARGET_SIZE)
