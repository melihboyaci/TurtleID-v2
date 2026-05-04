"""
head_detection.py — Kafa Profili Doğrulama Ajanı (HeadDetectionWorker)
=======================================================================

Pipeline'ın ikinci adımı. Kullanıcının yüklediği görselin zaten
kırpılmış bir kafa profili olduğu varsayılır. Bu worker bounding box,
crop veya OpenCV tabanlı tespit YAPMAZ; Gemini Vision API ile görselde
net bir deniz kaplumbağası yan kafa profili olup olmadığını doğrular.

BlackBoard Akışı:
    Okur  : query_image_path
    Yazar : head_crop (np.ndarray, RGB), head_confidence (float)

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Yalnızca "görselde kaplumbağa kafa profili var mı?" sorusuna
#        yanıt verir. Kırpma, tensör dönüşümü veya embedding üretimi
#        bu worker'ın sorumluluğu dışındadır.
# OCP  : Doğrulama prompt'u değiştirilerek farklı türler (kuş, balık)
#        için genişletilebilir; mevcut yapı değişmez.
# DIP  : Gemini model adı ve API anahtarı config.py'den alınır.
#        Supervisor'da yapılan genai.configure() çağrısı process-wide
#        geçerli olduğundan burada tekrar yapılmaz.
# ─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image

from agents import BaseWorker
from config import GEMINI_MODEL_NAME, LOG_DIR


class HeadDetectionWorker(BaseWorker):
    """
    Gemini Vision ile kafa profili doğrulaması yapan worker.

    Görselin bir deniz kaplumbağası kafa profili içerip içermediğini
    belirlemek için Gemini multimodal API'sine JSON formatında yanıt
    isteği gönderir.

    Attributes:
        MAX_PROMPT_SIZE: Gemini'ye gönderilecek görselin maksimum
            kenar uzunluğu (piksel). Büyük görseller küçültülür.
        vision_model: Gemini GenerativeModel istemcisi.
    """

    MAX_PROMPT_SIZE: int = 1024

    def __init__(self, blackboard) -> None:
        super().__init__(blackboard)
        self.vision_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def execute(self) -> bool:
        """
        Görseli okur, Gemini'ye gönderir ve doğrulama sonucunu yazar.

        Returns:
            True: Kafa profili doğrulandı; head_crop BlackBoard'a yazıldı.
            False: Görsel okunamadı veya kafa profili doğrulanamadı.
        """
        image = cv2.imread(self.bb.query_image_path)
        if image is None:
            self.bb.fail(self.name, "Görsel okunamadı.")
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_rgb.size == 0:
            self.bb.fail(self.name, "Boş görsel.")
            return False

        is_valid, reason = self._validate_head_profile(image_rgb)
        if not is_valid:
            self.bb.fail(self.name, f"Kafa profili doğrulanamadı: {reason}")
            return False

        os.makedirs(LOG_DIR, exist_ok=True)
        cv2.imwrite(
            os.path.join(LOG_DIR, "debug_head_crop.jpg"),
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        )

        self.bb.head_crop = image_rgb
        self.bb.head_confidence = 1.0
        self.log(f"Kafa profili doğrulandı: {reason}")
        return True

    def _validate_head_profile(self, image_rgb: np.ndarray) -> tuple[bool, str]:
        """
        Gemini Vision API ile kafa profili doğrulaması yapar.

        Args:
            image_rgb: (H, W, 3) şeklinde RGB numpy dizisi.

        Returns:
            (is_valid, reason) tuple'ı.
        """
        try:
            pil_img = self._prepare_prompt_image(image_rgb)
            prompt = (
                "GÖREV: Bu görselde bir DENİZ KAPLUMBAĞASI KAFA PROFİLİ var mı?\n\n"
                "Lütfen ÇOK ESNEK ve AFFEDİCİ ol. Bu görseller doğada çekildiği için bulanık, soluk veya karanlık olabilir. "
                "Eğer fotoğrafta bir kaplumbağa kafası görebiliyorsan kabul et.\n\n"
                "Kabul kriterleri (BİR TANESİ BİLE yeterlidir):\n"
                "- Görselde deniz kaplumbağasına ait bir kafa veya yüz yapısı görünüyorsa\n"
                "- Göz veya ağız tam net olmasa bile genel olarak bir kafa profili anlaşılabiliyorsa\n"
                "- Fotoğraf biraz bulanık veya karanlık olsa da kafa olduğu seçilebiliyorsa\n\n"
                "Reddetme nedenleri (SADECE şu durumlarda reddet):\n"
                "- Görselde kesinlikle kaplumbağa kafa/yüz parçası YOKSA (sadece kabuk, sadece kum, sadece su vb.)\n"
                "- Görsel kaplumbağa değil de tamamen alakasız başka bir nesneyse\n\n"
                "SADECE şu JSON formatında yanıt ver, başka metin yazma:\n"
                '{\n'
                '  "is_valid": true veya false,\n'
                '  "reason": "kısa Türkçe açıklama"\n'
                '}'
            )

            response = self.vision_model.generate_content([prompt, pil_img])
            text = (response.text or "").strip()
            data = self._parse_json_block(text)
            if data is None:
                return False, f"Gemini yanıtı ayrıştırılamadı: {text[:120]}"

            is_valid = bool(data.get("is_valid", False))
            reason = str(data.get("reason", "Açıklama yok.")).strip()
            return is_valid, reason
        except Exception as e:
            return False, f"Gemini API hatası: {e}"

    def _prepare_prompt_image(self, image_rgb: np.ndarray) -> Image.Image:
        """
        Görseli Gemini prompt'u için uygun boyuta küçültür.

        Args:
            image_rgb: (H, W, 3) şeklinde RGB numpy dizisi.

        Returns:
            Küçültülmüş PIL Image nesnesi.
        """
        pil_img = Image.fromarray(image_rgb)
        pil_img.thumbnail(
            (self.MAX_PROMPT_SIZE, self.MAX_PROMPT_SIZE),
            Image.LANCZOS,
        )
        return pil_img

    @staticmethod
    def _parse_json_block(text: str) -> Optional[dict]:
        """
        Gemini yanıtından JSON bloğunu ayrıştırır.

        Desteklenen formatlar:
            - Saf JSON: {"is_valid": true, ...}
            - Markdown code block: ```json ... ```
            - Karışık metin içinde gömülü JSON

        Args:
            text: Gemini'nin ham metin yanıtı.

        Returns:
            Ayrıştırılmış dict veya None.
        """
        if not text:
            return None

        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        candidate = match.group(1).strip() if match else text.strip()

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
