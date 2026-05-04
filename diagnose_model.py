"""
diagnose_model.py
-----------------
Migration sonrası turtle_embedding_model.keras dosyasının gerçekten
eğitilmiş Dense ağırlıkları içerip içermediğini kontrol eder.

1. h5 dosyasından Dense ağırlıklarını okur (h5py ile, Lambda bypass).
2. .keras dosyasındaki aynı Dense katmanının ağırlıklarıyla karşılaştırır.
3. Aynıysa migration başarılı, farklıysa başarısız.

Ayrıca aynı kaplumbağanın 2 görüntüsü ile farklı kaplumbağa arasında
cosine similarity karşılaştırması yapar.
"""

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

H5_PATH    = "turtle_embedding_model.h5"
KERAS_PATH = "turtle_embedding_model.keras"
DB_DIR     = "data/database"


def inspect_h5_weights(h5_path: str):
    """h5 dosyasındaki tüm katman isimlerini ve weight shape'lerini yazdırır."""
    print(f"\n{'='*60}")
    print(f"H5 INSPECT: {h5_path}")
    print('='*60)
    with h5py.File(h5_path, "r") as f:
        if "model_weights" in f:
            root = f["model_weights"]
        else:
            root = f
        dense_layers = []
        for layer_name in root.keys():
            layer_group = root[layer_name]
            if hasattr(layer_group, "keys"):
                for weight_name in layer_group.keys():
                    item = layer_group[weight_name]
                    if hasattr(item, "keys"):
                        for wn in item.keys():
                            w = item[wn]
                            if hasattr(w, "shape"):
                                if "dense" in layer_name.lower():
                                    dense_layers.append((layer_name, wn, w.shape, np.array(w)))
                                    print(f"  {layer_name}/{wn}: {w.shape}")
        return dense_layers


def compare_models():
    """H5 ve keras modellerindeki Dense ağırlıklarını karşılaştırır."""
    print("\n[1] H5 dosyasındaki Dense katmanları okunuyor...")
    h5_dense = inspect_h5_weights(H5_PATH)

    print("\n[2] .keras modeli yükleniyor...")
    keras_model = tf.keras.models.load_model(KERAS_PATH, compile=False)

    print("\n[3] .keras modelindeki Dense katmanları:")
    keras_dense = []
    for layer in keras_model.layers:
        if "dense" in layer.name.lower():
            for w in layer.get_weights():
                print(f"  {layer.name}: {w.shape}")
                keras_dense.append((layer.name, w))

    print("\n[4] Ağırlık karşılaştırması (h5 vs keras):")
    if len(h5_dense) == 0:
        print("  ⚠️ H5'te Dense ağırlığı bulunamadı!")
        return

    for (h5_layer, h5_wn, h5_shape, h5_w), (k_name, k_w) in zip(h5_dense, keras_dense):
        if h5_w.shape == k_w.shape:
            diff = np.abs(h5_w - k_w).mean()
            match = "✓ EŞİT" if diff < 1e-6 else f"✗ FARKLI (mean |diff| = {diff:.6f})"
            print(f"  {h5_layer}/{h5_wn} vs {k_name} [{h5_shape}]: {match}")
        else:
            print(f"  Shape mismatch: {h5_shape} vs {k_w.shape}")


def similarity_test():
    """Aynı/farklı kaplumbağa görüntüleri arasında cosine similarity testi."""
    print(f"\n{'='*60}")
    print("SIMILARITY TEST")
    print('='*60)

    model = tf.keras.models.load_model(KERAS_PATH, compile=False)

    def extract(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        batch = np.expand_dims(img, axis=0).astype(np.float32)
        return model.predict(preprocess_input(batch), verbose=0)[0]

    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_2_images(folder):
        path = os.path.join(DB_DIR, folder)
        if not os.path.isdir(path):
            return None
        imgs = [f for f in os.listdir(path) if f.lower().endswith(".jpg")]
        return [os.path.join(path, i) for i in imgs[:2]] if len(imgs) >= 2 else None

    pairs = {"t007": get_2_images("t007"), "t300": get_2_images("t300")}
    pairs = {k: v for k, v in pairs.items() if v}

    if not pairs:
        print("  ⚠️ t007 veya t300 klasöründe yeterli görsel yok.")
        return

    embeds = {k: [extract(p) for p in v] for k, v in pairs.items()}

    print("\nAynı-birey similarity (yüksek olmalı, ideal >0.9):")
    for k, embs in embeds.items():
        sim = cos_sim(embs[0], embs[1])
        print(f"  {k} içi: {sim:.4f}")

    if "t007" in embeds and "t300" in embeds:
        cross = cos_sim(embeds["t007"][0], embeds["t300"][0])
        print(f"\nFarklı-birey similarity (düşük olmalı, ideal <0.7):")
        print(f"  t007 vs t300: {cross:.4f}")

        print("\n📊 Yorum:")
        within_007 = cos_sim(embeds["t007"][0], embeds["t007"][1])
        if within_007 - cross < 0.1:
            print("  ❌ Model ayrım yapamıyor → Dense ağırlıkları eğitilmemiş olabilir!")
        elif within_007 > 0.85 and cross < 0.7:
            print("  ✅ Model sağlıklı ayrım yapıyor.")
        else:
            print("  ⚠️ Orta performans — daha fazla eğitim gerekebilir.")


if __name__ == "__main__":
    compare_models()
    similarity_test()
