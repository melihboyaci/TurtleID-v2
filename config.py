"""
config.py — TurtleID Merkezi Yapılandırma Modülü
=================================================

Tüm yapılandırma sabitleri tek bir yerde toplanmıştır.
Çevresel değişkenler (API anahtarları vb.) .env dosyasından,
sabitler ise bu modülden okunur.

Kullanım:
    from config import GEMINI_API_KEY, EMBEDDING_MODEL_PATH, ...

Tasarım Kararları:
    - Hardcoded değerler kod içine gömülmek yerine burada merkezileştirilmiştir.
    - API anahtarları .env dosyasından okunarak güvenlik sağlanır.
    - Yeni bir yapılandırma değeri eklendiğinde sadece bu dosya değişir.

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Bu modül yalnızca yapılandırma sabitleri sağlar.
# OCP  : Yeni sabitler eklenebilir; mevcut tüketiciler etkilenmez.
# DIP  : Diğer modüller somut değerlere değil, bu modüle bağımlıdır.
#        Model değiştirilmek istendiğinde yalnızca bu dosya güncellenir.
# DRY  : Gemini model adı, dosya yolları, eşik değerleri gibi tekrarlanan
#        sabitler tek bir kaynakta tutulur.
# ─────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
# API Yapılandırması
# ══════════════════════════════════════════════════════════════
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

# ══════════════════════════════════════════════════════════════
# Model Yapılandırması
# ══════════════════════════════════════════════════════════════
EMBEDDING_MODEL_PATH: str = "turtle_embedding_model.keras"
EMBEDDING_DIM: int = 256
TARGET_SIZE: tuple[int, int] = (224, 224)

# ══════════════════════════════════════════════════════════════
# Dosya Yolları
# ══════════════════════════════════════════════════════════════
DATABASE_DIR: str = "data/database"
QUERY_DIR: str = "data/query"
CACHE_FILE: str = os.path.join(DATABASE_DIR, "embeddings_cache.json")
LOG_DIR: str = "logs"
REPORT_FILE: str = "gelisim_raporu.md"

# ══════════════════════════════════════════════════════════════
# Eşleşme Eşik Değerleri
# ══════════════════════════════════════════════════════════════
MATCH_THRESHOLD: float = 0.85
POSSIBLE_THRESHOLD: float = 0.70

# ══════════════════════════════════════════════════════════════
# Audit (Girdi Doğrulama) Sabitleri
# ══════════════════════════════════════════════════════════════
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB: int = 10
MIN_IMAGE_SIZE_PX: int = 100
