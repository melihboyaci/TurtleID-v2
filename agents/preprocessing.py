import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from agents import BaseWorker

TARGET_SIZE = (224, 224)


def to_tensor(img_rgb: np.ndarray) -> np.ndarray:
    """
    RGB numpy görselini ResNet50'ye hazır (1, 224, 224, 3) tensöre dönüştürür.

    Modül düzeyinde tanımlanmıştır; hem PreprocessingWorker (query yolu)
    hem de RecognitionWorker (DB görselleri) aynı fonksiyonu import eder.
    Bu sayede tensör hazırlama mantığı tek bir yerde (DRY) yaşar.
    """
    if img_rgb.shape[:2] != TARGET_SIZE:
        img_rgb = cv2.resize(img_rgb, TARGET_SIZE)
    tensor = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    return preprocess_input(tensor)


class PreprocessingWorker(BaseWorker):
    """
    SRP — Tek Sorumluluk: "Model için Tensör Hazırlığı"

    Görüntü iyileştirme (CLAHE vb.) YAPMAZ.
    Blackboard akışı: head_crop → model_ready_tensor

    Adımlar:
    1. head_crop'u blackboard'dan oku (HeadDetectionWorker'ın RGB çıktısı)
    2. (224, 224) boyutuna getir
    3. (1, 224, 224, 3) şekline genişlet
    4. ResNet50 preprocess_input'tan geçir
    5. Hazır tensörü model_ready_tensor olarak yaz
    """

    def execute(self) -> bool:
        if self.bb.head_crop is None:
            self.bb.fail(self.name, "head_crop yok.")
            return False

        tensor = to_tensor(self.bb.head_crop)
        self.bb.model_ready_tensor = tensor
        self.log(f"Tensör hazırlandı: {tensor.shape}, dtype={tensor.dtype}")
        return True
