import os
import json
import datetime
from typing import Optional
import cv2
import numpy as np
import tensorflow as tf
from agents import BaseWorker
from agents.preprocessing import to_tensor

MODEL_PATH = "turtle_embedding_model.keras"
CACHE_FILE = "data/database/embeddings_cache.json"


class RecognitionWorker(BaseWorker):
    """
    Triplet Loss / Siamese Network ile eğitilmiş özel model üzerinden
    256 boyutlu L2-normalize embedding çıkarır.

    Bilimsel temel (Chabrolle & Dumont-Dayot, 2015):
    Her kaplumbağanın sağ VE sol profili ayrı ayrı
    embedding'e dönüştürülür. İkisinin ortalaması
    o bireyin "parmak izi" vektörü olur.
    """

    def __init__(self, blackboard):
        super().__init__(blackboard)
        self.log(f"Özel model yükleniyor: {MODEL_PATH}")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        self.log(f"Model hazır. Çıktı boyutu: {self.model.output_shape}")

    def execute(self) -> bool:
        """Query + DB embedding'lerini üretir, blackboard'a yazar."""
        # 1. Query embedding — PreprocessingWorker tarafından hazırlanmış tensörü kullan
        if self.bb.model_ready_tensor is None:
            self.bb.fail(self.name, "model_ready_tensor yok (PreprocessingWorker çalışmadı mı?)")
            return False
        self.bb.query_embedding = self.model.predict(self.bb.model_ready_tensor, verbose=0)[0]

        # 2. DB embeddings (sağ + sol profil ortalaması)
        db_dir = "data/database"
        turtle_folders = [
            f for f in os.listdir(db_dir)
            if os.path.isdir(os.path.join(db_dir, f))
        ]

        if not turtle_folders:
            self.bb.fail(self.name, "Veritabanında kayıtlı kaplumbağa yok.")
            return False

        cache         = self._load_cache()
        cache_updated = False

        self.bb.db_files = []
        self.bb.db_embeddings = []

        for folder in turtle_folders:
            folder_path = os.path.join(db_dir, folder)
            jpg_files   = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
            image_count = len(jpg_files)

            cached = cache.get(folder)
            # Yeni format: "embeddings" (çoğul) anahtarı + image_count
            if cached and cached.get("image_count") == image_count and "embeddings" in cached:
                embeddings = np.array(cached["embeddings"])
                self.log(f"{folder}: cache'den yüklendi ({image_count} görsel, {len(embeddings)} embedding).")
            else:
                embeddings = self._extract_all_embeddings(folder_path, jpg_files)
                if embeddings is not None and len(embeddings) > 0:
                    cache[folder] = {
                        "embeddings" : embeddings.tolist(),
                        "image_count": image_count,
                        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    }
                    cache_updated = True

            if embeddings is not None and len(embeddings) > 0:
                name = self._read_name(folder_path, folder)
                self.bb.db_files.append(name)
                self.bb.db_embeddings.append(embeddings)
                self.log(f"{name}: {len(embeddings)} embedding hazır.")

        if cache_updated:
            self._save_cache(cache)

        if not self.bb.db_embeddings:
            self.bb.fail(self.name, "Hiçbir DB kaydından embedding çıkarılamadı.")
            return False

        self.log(f"{len(self.bb.db_embeddings)} kaplumbağa için embedding üretildi.")
        return True

    def _extract_all_embeddings(self, folder_path: str, jpg_files: list) -> Optional[np.ndarray]:
        """
        Klasördeki tüm .jpg dosyaları için ayrı ayrı embedding çıkarır.
        Max-of-images yaklaşımı: averaging-blur problemini önlemek için
        ortalama almak yerine tüm embedding'leri döner.
        EvaluationWorker query ile her embedding'in benzerliğini hesaplayıp
        max'i alacak.
        """
        embeddings: list = []

        for filename in jpg_files:
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = self.model.predict(to_tensor(img_rgb), verbose=0)[0]
            embeddings.append(emb)

        if not embeddings:
            return None

        return np.array(embeddings)

    def _load_cache(self) -> dict:
        """embeddings_cache.json dosyasını yükler; yoksa boş dict döner."""
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: dict) -> None:
        """Cache dict'ini embeddings_cache.json dosyasına yazar."""
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        self.log("Embedding cache güncellendi.")

    def _read_name(self, folder_path: str, fallback: str) -> str:
        """metadata.json varsa ismi oradan okur."""
        meta_path = os.path.join(folder_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("name", fallback)
        return fallback.capitalize()

