import os
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from agents import BaseWorker


class HeadDetectionWorker(BaseWorker):
    """
    OpenCV ile kafa tespiti yapar, Gemini Vision ile doğrular.
    Sonucu blackboard.head_crop ve head_confidence'a yazar.
    """

    MIN_HEAD_RATIO = 0.05
    MAX_HEAD_RATIO = 0.80
    LOW_CONFIDENCE_THRESHOLD = 0.4
    FALLBACK_CROP_RATIO = 0.6

    def __init__(self, blackboard):
        super().__init__(blackboard)
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.vision_model = genai.GenerativeModel("gemini-1.5-flash")

    def execute(self) -> bool:
        image = cv2.imread(self.bb.query_image_path)
        if image is None:
            self.bb.fail(self.name, "Görsel okunamadı.")
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox, confidence = self._detect_with_contour(image_rgb)

        if bbox and confidence >= self.LOW_CONFIDENCE_THRESHOLD:
            cropped = self._crop_from_bbox(image_rgb, bbox)
        else:
            cropped = self._fallback_crop(image_rgb)

        # Gemini Vision ile doğrula
        verified, explanation = self._verify_with_gemini(cropped)
        self.log(f"Gemini: {explanation}")

        if not verified:
            self.bb.fail(self.name, f"Kafa doğrulanamadı: {explanation}")
            return False

        self.bb.head_crop = cropped
        self.bb.head_confidence = confidence
        self.log(f"Kafa tespiti başarılı. Güven: %{confidence*100:.0f}")
        return True

    def _verify_with_gemini(self, cropped: np.ndarray) -> tuple:
        try:
            pil_img = Image.fromarray(cropped)
            prompt = """Bu görselde bir deniz kaplumbağasının lateral (yan) kafa profili görünüyor mu?
            SONUÇ: EVET veya HAYIR
            AÇIKLAMA: (tek cümle Türkçe)"""
            response = self.vision_model.generate_content([prompt, pil_img])
            text = response.text.strip()
            verified = "SONUÇ: EVET" in text
            explanation = next(
                (l.replace("AÇIKLAMA:", "").strip()
                 for l in text.split("\n") if "AÇIKLAMA:" in l),
                "Analiz tamamlandı."
            )
            return verified, explanation
        except Exception as e:
            self.log(f"Gemini hatası (atlanıyor): {e}")
            return True, "Gemini doğrulaması atlandı."

    def _detect_with_contour(self, image: np.ndarray) -> tuple:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        total = image.shape[0] * image.shape[1]
        ratio = (w * h) / total
        if not (self.MIN_HEAD_RATIO <= ratio <= self.MAX_HEAD_RATIO):
            return None, 0.0
        return (x, y, w, h), min(1.0, ratio / self.MAX_HEAD_RATIO)

    def _fallback_crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        nh, nw = int(h * self.FALLBACK_CROP_RATIO), int(w * self.FALLBACK_CROP_RATIO)
        top, left = (h - nh) // 2, (w - nw) // 2
        return image[top:top+nh, left:left+nw]

    def _crop_from_bbox(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
