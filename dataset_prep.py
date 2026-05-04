"""
dataset_prep.py
---------------
annotations.json'daki kafa (head) koordinatlarını kullanarak ham görselleri
kırpar ve data/database/{turtle_id}/ altına organize eder.

Çalıştırma dizini: turtle-id/
    python dataset_prep.py
"""

import os
import json
import cv2
from collections import defaultdict

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Kaggle veri setinin içinde hem annotations.json hem images/ klasörü
# turtles-data/data/ altında yer alıyor.
RAW_DATA_DIR  = os.path.join("turtles-data", "data")
JSON_PATH     = os.path.join(RAW_DATA_DIR, "annotations.json")
OUTPUT_DB_DIR = os.path.join("data", "database")

PADDING        = 0.10   # bbox etrafına eklenen oran (0.10 = %10)
SKIP_OCCLUDED  = True   # occluded=True olan kafaları atla
SKIP_ISCROWD   = False  # Bu veri setinde iscrowd=1, RLE encoding anlamına gelir — atlanmamalı
LOG_INTERVAL   = 500    # kaç işlemde bir ilerleme yazılsın
# ──────────────────────────────────────────────────────────────────────────────


def load_json(path: str) -> dict:
    print(f"[INFO] JSON yükleniyor: {path}  (bu büyük dosya için birkaç saniye sürebilir)")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_map(images: list) -> dict:
    """image_id -> image_info (file_name, identity, width, height) eşlemesi"""
    return {img["id"]: img for img in images}


def get_head_annotations(annotations: list) -> list:
    """category_id == 3 (head) olan anotasyonları döndür."""
    return [a for a in annotations if a.get("category_id") == 3]


def crop_bbox(image, bbox: list, padding: float = 0.0):
    """
    COCO formatındaki bbox'ı [x, y, w, h] olarak alır,
    isteğe bağlı padding uygular ve kırpılmış bölgeyi döndürür.
    """
    x, y, w, h = [int(v) for v in bbox]
    ih, iw = image.shape[:2]

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)

    return image[y1:y2, x1:x2]


def main():
    data       = load_json(JSON_PATH)
    image_map  = build_image_map(data["images"])
    head_anns  = get_head_annotations(data["annotations"])

    print(f"[INFO] Toplam kafa anotasyonu : {len(head_anns)}")
    print(f"[INFO] Çıktı dizini           : {os.path.abspath(OUTPUT_DB_DIR)}")
    print()

    # (turtle_id, orientation) -> kaç dosya kaydedildi
    counters: dict = defaultdict(int)
    saved   = 0
    skipped = 0

    for idx, ann in enumerate(head_anns):
        attrs = ann.get("attributes", {})

        if SKIP_OCCLUDED and attrs.get("occluded", False):
            skipped += 1
            continue

        if SKIP_ISCROWD and ann.get("iscrowd", 0) == 1:
            skipped += 1
            continue

        img_info = image_map.get(ann["image_id"])
        if img_info is None:
            skipped += 1
            continue

        turtle_id   = img_info.get("identity", "unknown")
        file_name   = img_info["file_name"]          # örn: "images/t001/XYZ.JPG"
        orientation = attrs.get("orientation", "unknown").lower()

        img_path = os.path.join(RAW_DATA_DIR, file_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Görsel bulunamadı: {img_path}")
            skipped += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Görsel okunamadı : {img_path}")
            skipped += 1
            continue

        cropped = crop_bbox(image, ann["bbox"], PADDING)
        if cropped.size == 0:
            skipped += 1
            continue

        # Çıktı klasörünü oluştur
        out_dir = os.path.join(OUTPUT_DB_DIR, turtle_id)
        os.makedirs(out_dir, exist_ok=True)

        # Numaralı dosya adı: head_left_001.jpg
        key      = f"{turtle_id}_{orientation}"
        counters[key] += 1
        filename  = f"head_{orientation}_{counters[key]:03d}.jpg"
        out_path  = os.path.join(out_dir, filename)

        cv2.imwrite(out_path, cropped)
        saved += 1

        if saved % LOG_INTERVAL == 0:
            print(f"[INFO] {saved} kafa kaydedildi... (atlan: {skipped})")

    print()
    print("[DONE] İşlem tamamlandı.")
    print(f"  Kaydedilen : {saved}")
    print(f"  Atlanan    : {skipped}")
    print(f"  Çıktı dizini: {os.path.abspath(OUTPUT_DB_DIR)}")


if __name__ == "__main__":
    main()
