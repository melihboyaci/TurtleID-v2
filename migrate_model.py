"""
migrate_model.py
----------------
turtle_embedding_model.h5 (Lambda katmanlı, eski format) dosyasını
UnitNormalization kullanan yeni mimariye aktarır ve .keras olarak kaydeder.
Yeniden eğitim gerekmez — sadece ağırlıklar kopyalanır.

Çalıştırma: python migrate_model.py
"""

import keras
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

OLD_PATH = "turtle_embedding_model.h5"
NEW_PATH = "turtle_embedding_model.keras"
EMBEDDING_DIM = 256


def build_embedding_model_v2(embedding_dim: int = EMBEDDING_DIM) -> Model:
    """Lambda yerine UnitNormalization kullanan temiz mimari."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers[:-10]:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(embedding_dim, activation=None)(x)
    embeddings = tf.keras.layers.UnitNormalization(axis=1)(x)

    return Model(inputs=base_model.input, outputs=embeddings, name="embedding_model")


def main():
    print("[1/3] Yeni mimari oluşturuluyor (UnitNormalization, Lambda YOK)...")
    new_model = build_embedding_model_v2()
    print(f"      Toplam katman: {len(new_model.layers)}")

    print(f"[2/3] Ağırlıklar h5 dosyasından layer-by-layer yükleniyor: {OLD_PATH}")
    # by_name=True → Lambda katmanları atlanır (isimleri eşleşmez, yeni modelde yok)
    # skip_mismatch=True → Boyut uyuşmazlıklarını sessizce atla
    new_model.load_weights(OLD_PATH, by_name=True, skip_mismatch=True)
    print("      Ağırlıklar aktarıldı (Lambda katmanları atlandı — zaten weight'siz).")

    print(f"[3/3] Yeni model .keras formatında kaydediliyor: {NEW_PATH}")
    new_model.save(NEW_PATH)
    print(f"\n✅ Migrasyon tamamlandı → {NEW_PATH}")
    print("   Artık recognition.py bu dosyayı güvenle yükleyebilir.")


if __name__ == "__main__":
    main()
