import os
import json
from typing import Optional
import cv2
import numpy as np
from agents import BaseWorker
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


class RecognitionWorker(BaseWorker):
    """
    ResNet50 ile embedding çıkarır.

    Bilimsel temel (Chabrolle & Dumont-Dayot, 2015):
    Her kaplumbağanın sağ VE sol profili ayrı ayrı
    embedding'e dönüştürülür. İkisinin ortalaması
    o bireyin "parmak izi" vektörü olur.
    """

    TARGET_SIZE = (224, 224)

    def __init__(self, blackboard):
        super().__init__(blackboard)
        self.log("ResNet50 yükleniyor...")
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.log("Model hazır.")

    def execute(self) -> bool:
        """Query + DB embedding'lerini üretir, blackboard'a yazar."""
        # 1. Query embedding
        if self.bb.processed_image is None:
            self.bb.fail(self.name, "İşlenmiş görsel yok.")
            return False
        self.bb.query_embedding = self._extract(self.bb.processed_image)

        # 2. DB embeddings (sağ + sol profil ortalaması)
        db_dir = "data/database"
        turtle_folders = [
            f for f in os.listdir(db_dir)
            if os.path.isdir(os.path.join(db_dir, f))
        ]

        if not turtle_folders:
            self.bb.fail(self.name, "Veritabanında kayıtlı kaplumbağa yok.")
            return False

        self.bb.db_files = []
        self.bb.db_embeddings = []

        for folder in turtle_folders:
            folder_path = os.path.join(db_dir, folder)
            embedding = self._extract_dual_profile(folder_path)

            if embedding is not None:
                name = self._read_name(folder_path, folder)
                self.bb.db_files.append(name)
                self.bb.db_embeddings.append(embedding)
                self.log(f"{name}: sağ+sol profil embedding hazır.")

        if not self.bb.db_embeddings:
            self.bb.fail(self.name, "Hiçbir DB kaydından embedding çıkarılamadı.")
            return False

        self.log(f"{len(self.bb.db_embeddings)} kaplumbağa için embedding üretildi.")
        return True

    def _extract_dual_profile(self, folder_path: str) -> Optional[np.ndarray]:
        """
        Sağ ve sol profil görselinden embedding çıkarır.
        İkisinin ortalamasını döner.
        Sadece biri varsa onu kullanır.
        İkisi de yoksa None döner.
        """
        sag = os.path.join(folder_path, "sag_profil.jpg")
        sol = os.path.join(folder_path, "sol_profil.jpg")

        embeddings = []

        for profile_path in [sag, sol]:
            if os.path.exists(profile_path):
                img = cv2.imread(profile_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, self.TARGET_SIZE)
                    embeddings.append(self._extract(img_resized))

        if not embeddings:
            return None

        # İki profilin ortalaması = bireyin parmak izi
        return np.mean(embeddings, axis=0)

    def _read_name(self, folder_path: str, fallback: str) -> str:
        """metadata.json varsa ismi oradan okur."""
        meta_path = os.path.join(folder_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("name", fallback)
        return fallback.capitalize()

    def _extract(self, image: np.ndarray) -> np.ndarray:
        """Tek bir görselden ResNet50 embedding çıkarır."""
        batch = np.expand_dims(image, axis=0).astype(np.float32)
        preprocessed = preprocess_input(batch)
        return self.model.predict(preprocessed, verbose=0)[0]
