import os
from PIL import Image
from agents import BaseWorker


class AuditWorker(BaseWorker):
    """Girdi dosyasını doğrular. Blackboard'a audit_result yazar."""

    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    MAX_FILE_SIZE_MB = 10
    MIN_IMAGE_SIZE_PX = 100

    def execute(self) -> bool:
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

    def _check_extension(self, path: str) -> tuple:
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"Desteklenmeyen format: {ext}"
        return True, "OK"

    def _check_size(self, path: str) -> tuple:
        if not os.path.exists(path):
            return False, f"Dosya bulunamadı: {path}"
        mb = os.path.getsize(path) / (1024 * 1024)
        if mb > self.MAX_FILE_SIZE_MB:
            return False, f"Dosya çok büyük: {mb:.1f}MB"
        return True, "OK"

    def _check_readable(self, path: str) -> tuple:
        try:
            with Image.open(path) as img:
                img.verify()
            return True, "OK"
        except Exception as e:
            return False, f"Görsel okunamadı: {e}"

    def _check_dimensions(self, path: str) -> tuple:
        with Image.open(path) as img:
            w, h = img.size
        if w < self.MIN_IMAGE_SIZE_PX or h < self.MIN_IMAGE_SIZE_PX:
            return False, f"Görsel çok küçük: {w}x{h}px"
        return True, "OK"
