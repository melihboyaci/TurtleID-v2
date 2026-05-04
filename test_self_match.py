"""
test_self_match.py
------------------
DB'deki bir kaplumbağa klasöründen rastgele görüntü seçip,
DB prototype'ları ile karşılaştırır.

BEKLENEN: kendi sınıfı #1 olmalı (%90+).
Eğer kendi sınıfı #1 değilse → prototype hatalı.
Eğer #1 ama düşük skor (~%70-80) → prototype zayıf.
"""

import os
import json
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.spatial.distance import cosine

MODEL_PATH = "turtle_embedding_model.keras"
CACHE_FILE = "data/database/embeddings_cache.json"
DB_DIR     = "data/database"
TARGET_SIZE = (224, 224)


def extract(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    batch = np.expand_dims(img, axis=0).astype(np.float32)
    return model.predict(preprocess_input(batch), verbose=0)[0]


def test_turtle(model, cache, turtle_id):
    folder = os.path.join(DB_DIR, turtle_id)
    if not os.path.isdir(folder):
        print(f"\n❌ {turtle_id} klasörü yok")
        return

    imgs = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    print(f"\n{'='*60}")
    print(f"TEST: {turtle_id} ({len(imgs)} görsel)")
    print('='*60)
    for img in imgs[:10]:
        print(f"  - {img}")
    if len(imgs) > 10:
        print(f"  ... +{len(imgs)-10} dosya")

    if not imgs:
        return

    # Rastgele bir görüntü seç ve query yap
    test_img = random.choice(imgs)
    query_path = os.path.join(folder, test_img)
    print(f"\n🎯 Query: {test_img}")

    query_emb = extract(model, query_path)

    # Tüm DB ile karşılaştır (max-of-images)
    scores = []
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    for tid, data in cache.items():
        # Yeni format: "embeddings" listesi
        if "embeddings" in data:
            db_arr = np.array(data["embeddings"])
        else:
            # Geriye uyumluluk
            db_arr = np.array(data["embedding"])[np.newaxis, :]
        db_norms = db_arr / (np.linalg.norm(db_arr, axis=1, keepdims=True) + 1e-12)
        sims = db_norms @ q_norm
        scores.append((tid, float(sims.max())))

    scores.sort(key=lambda x: x[1], reverse=True)

    print("\nTOP-10:")
    for rank, (tid, sim) in enumerate(scores[:10], 1):
        marker = "  👈 KENDİ SINIFI" if tid.lower() == turtle_id.lower() else ""
        print(f"  #{rank}: {tid:8s} | %{sim*100:.2f}{marker}")

    # Kendi sınıfı kaçıncı sırada?
    own_rank = next((r for r, (tid, _) in enumerate(scores, 1) if tid.lower() == turtle_id.lower()), None)
    own_score = next((s for tid, s in scores if tid.lower() == turtle_id.lower()), None)

    if own_rank == 1:
        print(f"\n✅ Kendi sınıfı #1 (%{own_score*100:.2f}) — model sağlıklı.")
    elif own_rank:
        print(f"\n⚠️ Kendi sınıfı #{own_rank}. sırada (%{own_score*100:.2f}) — prototype zayıf.")
    else:
        print(f"\n❌ Kendi sınıfı listede yok!")


def main():
    print("Model yükleniyor...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print(f"Cache yükleniyor: {CACHE_FILE}")
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
    print(f"Cache'de {len(cache)} kaplumbağa.")

    # T007 ve T300'ü test et
    for tid in ["t007", "t300", "t426", "t422"]:
        test_turtle(model, cache, tid)


if __name__ == "__main__":
    main()
