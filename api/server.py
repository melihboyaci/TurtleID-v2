"""
api/server.py — TurtleID REST API
==================================

Mevcut agent pipeline'ını HTTP üzerinden erişilebilir kılar.
FastAPI backend; tanımlama, kayıt ve veritabanı listeleme
endpoint'lerini sağlar.

Çalıştırma:
    python run_server.py          (proje kökünden)
    http://localhost:8000         (tarayıcıdan)
"""

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

# ── Proje kök dizinini sys.path'e ekle ──────────────────────────────────────
# Bu dosya turtle-id/api/ altında yaşar; agents/, blackboard.py, config.py
# ise turtle-id/ altındadır. Import'ların çalışması için kök eklenir.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.supervisor import SupervisorAgent
from config import DATABASE_DIR, EMBEDDING_MODEL_PATH
import kayit_yardimcisi

# ── Sabitler ────────────────────────────────────────────────────────────────────────────
IMAGE_EXTS: frozenset[str]    = frozenset({".jpg", ".jpeg", ".png"})
VALID_PROFILES: frozenset[str] = frozenset({"sag", "sol"})

app = FastAPI(
    title="TurtleID API",
    description="Deniz kaplumbağası kimlik tespit sistemi REST API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model yükleme durumu ─────────────────────────────────────────────────────────────────────────
_model_loaded: bool = Path(_PROJECT_ROOT / EMBEDDING_MODEL_PATH).exists()


# ════════════════════════════════════════════════════════════════════════════
# YARDIMCI CONTEXT MANAGER'LAR
# ════════════════════════════════════════════════════════════════════════════

@contextlib.asynccontextmanager
async def _temp_upload(upload: UploadFile):
    """Yüklenen dosyayı geçici diske yazar; blok çıkışında siler."""
    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await upload.read())
        tmp_path = tmp.name
    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@contextlib.contextmanager
def _project_cwd():
    """Blok süresince cwd'yi proje köküne alır, çıkışta geri döndürür."""
    original = os.getcwd()
    os.chdir(str(_PROJECT_ROOT))
    try:
        yield
    finally:
        os.chdir(original)


@contextlib.contextmanager
def _muted_stdin():
    """Blok süresince sys.stdin'i boş StringIO ile maskeler.

    kayit_yardimcisi.detect_species() Gemini hatası durumunda input() çağırır;
    bu context manager, API ortamında o çağırının bloke etmesini önler.
    """
    original = sys.stdin
    sys.stdin = io.StringIO("")
    try:
        yield
    finally:
        sys.stdin = original


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINT — GET /health
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", summary="Sistem sağlık kontrolü")
async def health() -> dict:
    """Model yükleme durumunu ve API sağlığını döndürür."""
    return {"status": "ok", "model_loaded": _model_loaded}


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINT — GET /api/turtles
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/turtles", summary="Kayıtlı kaplumbağaları listele")
async def list_turtles() -> dict:
    """
    data/database/ klasörünü tarar.

    Her alt klasördeki metadata.json ve görsel sayısını döndürür.

    Returns:
        count: Toplam kayıtlı birey sayısı.
        turtles: Birey bilgilerinin listesi.
    """
    db_path = _PROJECT_ROOT / DATABASE_DIR

    if not db_path.exists():
        return {"count": 0, "turtles": []}

    turtles: list[dict] = []
    try:
        for folder in sorted(db_path.iterdir()):
            if not folder.is_dir():
                continue

            meta_file = folder / "metadata.json"
            if not meta_file.exists():
                continue

            with meta_file.open("r", encoding="utf-8") as f:
                meta: dict = json.load(f)

            image_count = sum(
                1 for p in folder.iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            )

            turtles.append(
                {
                    "id": meta.get("id", folder.name),
                    "name": meta.get("name", folder.name),
                    "species": meta.get("species", "Bilinmiyor"),
                    "image_count": image_count,
                }
            )

        return {"count": len(turtles), "turtles": turtles}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Veritabanı okunamadı: {exc}") from exc


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINT — POST /api/identify
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/identify", summary="Kaplumbağa kimlik tespiti")
async def identify(image: UploadFile = File(...)) -> JSONResponse:
    """
    Yüklenen kafa profilini pipeline üzerinden kimlik tespitine gönderir.

    Args:
        image: Kaplumbağa kafa profili görseli (JPG/PNG, max 10 MB).

    Returns:
        success, identity, score, status, mission_log, error alanlarını
        içeren JSON yanıtı.
    """
    try:
        async with _temp_upload(image) as tmp_path:
            with _project_cwd():
                supervisor = SupervisorAgent(image_path=tmp_path)
                blackboard = supervisor.run_mission()

        match   = blackboard.match_result
        success = blackboard.mission_status == "SUCCESS"

        return JSONResponse(
            content={
                "success":     success,
                "identity":    match.get("name") if success else None,
                "score":       round(match.get("score", 0.0) * 100, 1) if success else None,
                "status":      match.get("status") if success else None,
                "mission_log": blackboard.mission_log,
                "error":       blackboard.error_message if not success else None,
            }
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False, "identity": None, "score": None,
                "status": None,   "mission_log": [], "error": f"Sunucu hatası: {exc}",
            },
        )


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINT — POST /api/register
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/register", summary="Yeni kaplumbağa kaydı")
async def register(
    image: UploadFile = File(...),
    name: str = Form(...),
    profile: str = Form(...),
    notes: str = Form(""),
) -> JSONResponse:
    """
    Kaplumbağa kafa görselini veritabanına kaydeder.

    Gemini Vision ile tür tespiti yapılır; kayit_yardimcisi.register_turtle()
    çağrılır. Gemini erişilemezse stdin yönlendirmesi ile varsayılan değerler
    kullanılır (input() çağrısı boş string döner).

    Args:
        image:   Kafa profili görseli.
        name:    Kaplumbağa adı / etiketi.
        profile: Profil tarafı — "sag" veya "sol".
        notes:   Ek not (opsiyonel).

    Returns:
        success, message, species alanlarını içeren JSON yanıtı.
    """
    if profile not in VALID_PROFILES:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Profil değeri 'sag' veya 'sol' olmalıdır.",
                "species": None,
            },
        )

    try:
        async with _temp_upload(image) as tmp_path:
            with _muted_stdin(), _project_cwd():
                species_info = kayit_yardimcisi.detect_species(tmp_path)
                ok           = kayit_yardimcisi.register_turtle(tmp_path, name, profile, notes)

        if ok:
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"{name} başarıyla kaydedildi.",
                    "species": species_info.get("species_latin", "Bilinmiyor"),
                }
            )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Kayıt sırasında beklenmeyen bir hata oluştu.",
                "species": None,
            },
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Sunucu hatası: {exc}",
                "species": None,
            },
        )


# ════════════════════════════════════════════════════════════════════════════
# STATIC FILE SERVING — /  (frontend/)
# API route'ları önce tanımlanır; mount en sona gelir.
# ════════════════════════════════════════════════════════════════════════════

_frontend_dir = _PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
