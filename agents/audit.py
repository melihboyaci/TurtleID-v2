"""
audit.py — Girdi Dosyası Doğrulama Ajanı (AuditWorker)
========================================================

Pipeline'ın ilk adımı. Kullanıcının verdiği sorgu görselinin
format, boyut, okunabilirlik ve çözünürlük açısından geçerli
olduğunu doğrular.

BlackBoard Akışı:
    Okur  : query_image_path
    Yazar : audit_result  →  {"passed": bool, "message": str}

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Yalnızca girdi doğrulama yapar; görüntü işleme veya
#        model çağrısı bu worker'ın sorumluluğu dışındadır.
# OCP  : Yeni bir doğrulama kuralı eklemek için _check_* metodu
#        yazılıp execute()'deki listeye eklenir; mevcut kontroller
#        değişmez.
# DIP  : Doğrulama sabitleri (MAX_FILE_SIZE_MB vb.) merkezi
#        config.py modülünden alınır.
# Strategy Pattern: Her _check_* metodu bağımsız bir doğrulama
#        stratejisidir; execute() bunları sırayla çalıştırır.
# ─────────────────────────────────────────────────────────────
"""

import os

from PIL import Image

from agents import BaseWorker
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MIN_IMAGE_SIZE_PX


class AuditWorker(BaseWorker):
    """
    Sorgu görselinin pipeline'a girmeden önce temel geçerlilik
    kontrollerinden geçmesini sağlar.

    Doğrulama Zinciri:
        1. Dosya uzantısı kontrolü (.jpg, .jpeg, .png)
        2. Dosya boyutu kontrolü (≤ MAX_FILE_SIZE_MB)
        3. Görsel okunabilirlik kontrolü (PIL verify)
        4. Minimum çözünürlük kontrolü (≥ MIN_IMAGE_SIZE_PX)

    Herhangi bir kontrol başarısız olursa pipeline durdurulur.
    """

    def execute(self) -> bool:
        """
        Sıralı doğrulama zincirini çalıştırır.

        Returns:
            True: Tüm kontroller geçti; audit_result["passed"] = True.
            False: Bir kontrol başarısız oldu; hata BlackBoard'a yazıldı.
        """
        path = self.bb.query_image_path

        for check in [self._check_extension, self._check_size,
                      self._check_readable, self._check_dimensions]:
            passed, msg = check(path)
            if not passed:
                self.bb.audit_result = {"passed": False, "message": msg}
                self.bb.fail(self.name, msg)
                return False

        self.bb.audit_result = {"passed": True, "message": "Doğrulama başarılı."}
        self.log("Görsel doğrulama başarılı.")
        return True

    # ── Doğrulama Stratejileri ────────────────────────────────

    def _check_extension(self, path: str) -> tuple[bool, str]:
        """Dosya uzantısının desteklenen formatlardan biri olduğunu doğrular."""
        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Desteklenmeyen format: {ext}"
        return True, "OK"

    def _check_size(self, path: str) -> tuple[bool, str]:
        """Dosyanın var olduğunu ve boyut sınırını aşmadığını doğrular."""
        if not os.path.exists(path):
            return False, f"Dosya bulunamadı: {path}"
        mb = os.path.getsize(path) / (1024 * 1024)
        if mb > MAX_FILE_SIZE_MB:
            return False, f"Dosya çok büyük: {mb:.1f}MB"
        return True, "OK"

    def _check_readable(self, path: str) -> tuple[bool, str]:
        """Görselin PIL tarafından açılabildiğini ve bütünlüğünü doğrular."""
        try:
            with Image.open(path) as img:
                img.verify()
            return True, "OK"
        except Exception as e:
            return False, f"Görsel okunamadı: {e}"

    def _check_dimensions(self, path: str) -> tuple[bool, str]:
        """Görselin minimum çözünürlük gereksinimini karşıladığını doğrular."""
        with Image.open(path) as img:
            w, h = img.size
        if w < MIN_IMAGE_SIZE_PX or h < MIN_IMAGE_SIZE_PX:
            return False, f"Görsel çok küçük: {w}x{h}px"
        return True, "OK"
