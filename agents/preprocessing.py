"""
preprocessing.py — Tensör Hazırlık Ajanı (PreprocessingWorker)
===============================================================

Pipeline'ın üçüncü adımı. HeadDetectionWorker tarafından doğrulanmış
RGB kafa görselini, embedding modeline hazır tensör formatına dönüştürür.

BlackBoard Akışı:
    Okur  : head_crop (np.ndarray, RGB)
    Yazar : model_ready_tensor (np.ndarray, shape=(1, 224, 224, 3))

Bu worker, su altı görsellerindeki doku kayıplarını ve aydınlatma
farklılıklarını gidermek için CLAHE (Contrast Limited Adaptive Histogram
Equalization) uygular ve tensör formatına dönüştürür.

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Yalnızca "BlackBoard'dan RGB oku → iyileştir → tensöre çevir → geri yaz"
#        işini yapar. Model çağrısı yoktur.
# DRY  : Tensör dönüşüm mantığı agents/tensor_utils.py'de yaşar.
#        Bu worker ve RecognitionWorker aynı fonksiyonu kullanır.
# DIP  : to_tensor() fonksiyonu bağımsız bir utility modülünden gelir;
#        worker'lar arası doğrudan bağımlılık yoktur.
# ─────────────────────────────────────────────────────────────
"""

from agents import BaseWorker
from agents.tensor_utils import to_tensor


class PreprocessingWorker(BaseWorker):
    """
    Kafa görselini (RGB numpy dizisi) modele hazır tensöre dönüştürür.

    Dönüşüm Adımları:
        1. BlackBoard'dan ``head_crop`` alanını okur.
        2. ``to_tensor()`` ile TARGET_SIZE'a ölçekler, **CLAHE uygular**, batch boyutu ekler
           ve ResNet50 preprocess_input uygular.
        3. Sonucu ``model_ready_tensor`` olarak BlackBoard'a yazar.
    """

    def execute(self) -> bool:
        """
        Tensör hazırlığını gerçekleştirir.

        Returns:
            True: Tensör başarıyla hazırlandı ve BlackBoard'a yazıldı.
            False: head_crop bulunamadı (önceki ajan başarısız).
        """
        if self.bb.head_crop is None:
            self.bb.fail(self.name, "head_crop yok.")
            return False

        tensor = to_tensor(self.bb.head_crop)
        self.bb.model_ready_tensor = tensor
        self.log(f"Tensör hazırlandı: {tensor.shape}, dtype={tensor.dtype}")
        return True
