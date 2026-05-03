import json
import os
import re
from typing import Optional, Tuple

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from agents import BaseWorker


class HeadDetectionWorker(BaseWorker):
    """
    Head validation worker.

    Kullanıcının yüklediği görselin zaten kırpılmış kafa fotoğrafı olduğu
    varsayılır. Bu worker bounding box, crop veya OpenCV tabanlı tespit
    yapmaz; sadece Gemini'ye görselde net bir deniz kaplumbağası yan kafa
    profili olup olmadığını sorar.
    """

    MAX_PROMPT_SIZE = 1024

    def __init__(self, blackboard):
        super().__init__(blackboard)
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.vision_model = genai.GenerativeModel("gemini-2.5-flash")

    def execute(self) -> bool:
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

        os.makedirs("logs", exist_ok=True)
        cv2.imwrite(
            "logs/debug_head_crop.jpg",
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        )

        self.bb.head_crop = image_rgb
        self.bb.head_confidence = 1.0
        self.log(f"Kafa profili doğrulandı: {reason}")
        return True

    def _validate_head_profile(self, image_rgb: np.ndarray) -> Tuple[bool, str]:
        try:
            pil_img = self._prepare_prompt_image(image_rgb)
            prompt = (
                "GÖREV: Bu görselde net bir şekilde DENİZ KAPLUMBAĞASI "
                "KAFA PROFİLİ var mı?\n\n"
                "Kabul kriterleri (HEPSİ varsa Evet):\n"
                "- Görsel kırpılmış kafa fotoğrafı gibi görünür\n"
                "- LATERAL (yan) görünüm vardır: sağ veya sol yan profil\n"
                "- Göz net görünür\n"
                "- Gaga/ağız net görünür\n"
                "- Kafa veya boyun kabuktan/gövdeden ayırt edilebilir\n\n"
                "Reddetme nedenleri (BİRİ varsa Hayır):\n"
                "- Görselde kafa yoktur\n"
                "- Yüzgeç/palet, kabuk veya gövde ana odaktır\n"
                "- Ön, arka veya üst görünüm vardır; yan profil değildir\n"
                "- Görsel çok bulanık, çok karanlık veya kafa detayı belirsizdir\n"
                "- Bu bir deniz kaplumbağası kafa profili değildir\n\n"
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
        pil_img = Image.fromarray(image_rgb)
        pil_img.thumbnail(
            (self.MAX_PROMPT_SIZE, self.MAX_PROMPT_SIZE),
            Image.LANCZOS,
        )
        return pil_img

    @staticmethod
    def _parse_json_block(text: str) -> Optional[dict]:
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
