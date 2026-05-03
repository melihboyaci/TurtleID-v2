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
        return cv2.resize(image, self.TARGET_SIZE)
