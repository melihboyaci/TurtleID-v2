"""
recognition.py — Embedding Çıkarma Ajanı (RecognitionWorker)
=============================================================

Pipeline'ın dördüncü adımı. Triplet Loss / Siamese Network ile
eğitilmiş özel modeli kullanarak hem sorgu görselinden hem de
veritabanındaki tüm bireylerin görsellerinden 256 boyutlu
L2-normalize embedding vektörleri çıkarır.

Bilimsel Temel:
    Chabrolle & Dumont-Dayot (2015) — Her kaplumbağanın birden fazla
    profil görseli ayrı ayrı embedding'e dönüştürülür. EvaluationWorker
    bunları max-of-images yaklaşımıyla değerlendirir.

BlackBoard Akışı:
    Okur  : model_ready_tensor (query tensörü)
    Yazar : query_embedding, db_embeddings, db_files

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Bu worker "modeli çağırıp embedding çıkarma" işini yapar.
#        Cache I/O ve klasör tarama yardımcı private metotlardır;
#        ana sorumluluk embedding üretmektir.
#        Not: Tam SRP uyumu için cache yönetimi ayrı bir sınıfa
#        çıkarılabilirdi. Ancak ödev kapsamında bu trade-off bilinçli
#        olarak kabul edilmiştir (YAGNI — You Aren't Gonna Need It).
# DRY  : to_tensor() fonksiyonu agents/tensor_utils.py'den alınır;
#        PreprocessingWorker ile aynı dönüşüm mantığı paylaşılır.
# OCP  : Model değiştirilmek istendiğinde yalnızca config.py'deki
#        EMBEDDING_MODEL_PATH güncellenir; bu dosya değişmez.
# DIP  : Dosya yolları ve yapılandırma config.py'den okunur.
# ─────────────────────────────────────────────────────────────
"""

import os
import json
import datetime
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

from agents import BaseWorker
from agents.tensor_utils import to_tensor
from config import EMBEDDING_MODEL_PATH, CACHE_FILE, DATABASE_DIR


class RecognitionWorker(BaseWorker):
    """
    Triplet Loss modeli üzerinden embedding vektörleri üretir.

    ``__init__`` sırasında model dosyası yüklenir ve pipeline boyunca
    tekrar kullanılır. Veritabanı embedding'leri JSON tabanlı cache
    mekanizmasıyla hızlandırılır.

    Attributes:
        model: Yüklenmiş TensorFlow/Keras embedding modeli.
    """

    def __init__(self, blackboard) -> None:
        super().__init__(blackboard)
        self.log(f"Özel model yükleniyor: {EMBEDDING_MODEL_PATH}")
        self.model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, compile=False)
        self.log(f"Model hazır. Çıktı boyutu: {self.model.output_shape}")

    def execute(self) -> bool:
        """
        Query ve veritabanı embedding'lerini üretir, BlackBoard'a yazar.

        Returns:
            True: Tüm embedding'ler başarıyla üretildi.
            False: Tensör eksik veya veritabanı boş.
        """
        # 1. Query embedding — PreprocessingWorker tarafından hazırlanmış tensörü kullan
        if self.bb.model_ready_tensor is None:
            self.bb.fail(self.name, "model_ready_tensor yok (PreprocessingWorker çalışmadı mı?)")
            return False
        self.bb.query_embedding = self.model.predict(self.bb.model_ready_tensor, verbose=0)[0]

        # 2. Veritabanı embedding'leri (max-of-images yaklaşımı)
        turtle_folders = [
            f for f in os.listdir(DATABASE_DIR)
            if os.path.isdir(os.path.join(DATABASE_DIR, f))
        ]

        if not turtle_folders:
            self.bb.fail(self.name, "Veritabanında kayıtlı kaplumbağa yok.")
            return False

        cache         = self._load_cache()
        cache_updated = False

        self.bb.db_files = []
        self.bb.db_embeddings = []

        for folder in turtle_folders:
            folder_path = os.path.join(DATABASE_DIR, folder)
            jpg_files   = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
            image_count = len(jpg_files)

            cached = cache.get(folder)
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

    # ── Private Yardımcı Metotlar ─────────────────────────────

    def _extract_all_embeddings(self, folder_path: str, jpg_files: list) -> Optional[np.ndarray]:
        """
        Klasördeki tüm .jpg dosyaları için ayrı ayrı embedding çıkarır.

        Max-of-images yaklaşımı: averaging-blur problemini önlemek için
        ortalama almak yerine tüm embedding'ler döndürülür.
        EvaluationWorker query ile her embedding'in benzerliğini
        hesaplayıp max'i alır.

        Args:
            folder_path: Kaplumbağa klasörünün tam yolu.
            jpg_files: Klasördeki .jpg dosya adları listesi.

        Returns:
            (N, 256) şeklinde embedding matrisi veya None.
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
        """
        Embedding cache dosyasını yükler.

        Returns:
            Cache dict'i; dosya yoksa boş dict.
        """
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: dict) -> None:
        """
        Cache dict'ini JSON dosyasına yazar.

        Args:
            cache: Güncellenmiş cache verisi.
        """
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        self.log("Embedding cache güncellendi.")

    def _read_name(self, folder_path: str, fallback: str) -> str:
        """
        Kaplumbağa klasöründeki metadata.json'dan birey ismini okur.

        Args:
            folder_path: Kaplumbağa klasörünün tam yolu.
            fallback: metadata.json yoksa kullanılacak varsayılan isim.

        Returns:
            Birey ismi (str).
        """
        meta_path = os.path.join(folder_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("name", fallback)
        return fallback.capitalize()
