import os
import json
import datetime
from typing import Optional
import cv2
import numpy as np
from agents import BaseWorker
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

CACHE_FILE     = "data/database/embeddings_cache.json"
LEFT_PATTERNS  = ["head_left", "head_topleft", "sol_profil"]
RIGHT_PATTERNS = ["head_right", "head_topright", "sag_profil"]


class RecognitionWorker(BaseWorker):
    """
    ResNet50 ile embedding çıkarır.

    Bilimsel temel (Chabrolle & Dumont-Dayot, 2015):
    Her kaplumbağanın sağ VE sol profili ayrı ayrı
    embedding'e dönüştürülür. İkisinin ortalaması
    o bireyin "parmak izi" vektörü olur.
    """

    TARGET_SIZE       = (224, 224)
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: tuple  = (8, 8)
    SHARPEN_STRENGTH: float = 0.3

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

        cache         = self._load_cache()
        cache_updated = False

        self.bb.db_files = []
        self.bb.db_embeddings = []

        for folder in turtle_folders:
            folder_path = os.path.join(db_dir, folder)
            jpg_files   = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
            image_count = len(jpg_files)

            cached = cache.get(folder)
            if cached and cached.get("image_count") == image_count:
                embedding = np.array(cached["embedding"])
                self.log(f"{folder}: cache'den yüklendi ({image_count} görsel).")
            else:
                embedding = self._extract_dual_profile(folder_path, jpg_files)
                if embedding is not None:
                    cache[folder] = {
                        "embedding"  : embedding.tolist(),
                        "image_count": image_count,
                        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    }
                    cache_updated = True

            if embedding is not None:
                name = self._read_name(folder_path, folder)
                self.bb.db_files.append(name)
                self.bb.db_embeddings.append(embedding)
                self.log(f"{name}: embedding hazır.")

        if cache_updated:
            self._save_cache(cache)

        if not self.bb.db_embeddings:
            self.bb.fail(self.name, "Hiçbir DB kaydından embedding çıkarılamadı.")
            return False

        self.log(f"{len(self.bb.db_embeddings)} kaplumbağa için embedding üretildi.")
        return True

    def _extract_dual_profile(self, folder_path: str, jpg_files: list) -> Optional[np.ndarray]:
        """
        Klasördeki .jpg dosyalarını LEFT_PATTERNS / RIGHT_PATTERNS'a göre gruplar.
        Her grubun embedding ortalamasını alır; iki grup ortalamasının
        ortalamasını döner. Hiçbir pattern'a uymayan dosyalar (ör: head_top) atlanır.
        En az bir grup doluysa sonuç üretilir; her ikisi de boşsa None döner.
        """
        left_embeddings:  list = []
        right_embeddings: list = []

        for filename in jpg_files:
            fname_lower = filename.lower()
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is None:
                continue

            img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb     = self._enhance(img_rgb)
            img_resized = cv2.resize(img_rgb, self.TARGET_SIZE)
            emb         = self._extract(img_resized)

            if any(p in fname_lower for p in LEFT_PATTERNS):
                left_embeddings.append(emb)
            elif any(p in fname_lower for p in RIGHT_PATTERNS):
                right_embeddings.append(emb)
            # else: head_top gibi eşleşmeyen dosyalar atlanır

        group_means = []
        if left_embeddings:
            group_means.append(np.mean(left_embeddings, axis=0))
        if right_embeddings:
            group_means.append(np.mean(right_embeddings, axis=0))

        if not group_means:
            return None

        # Sol ve sağ grup ortalamalarının ortalaması = bireyin parmak izi
        return np.mean(group_means, axis=0)

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing ile aynı CLAHE pipeline."""
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

    def _extract(self, image: np.ndarray) -> np.ndarray:
        """Tek bir görselden ResNet50 embedding çıkarır."""
        batch = np.expand_dims(image, axis=0).astype(np.float32)
        preprocessed = preprocess_input(batch)
        return self.model.predict(preprocessed, verbose=0)[0]
