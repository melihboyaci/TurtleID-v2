"""
supervisor.py — Hiyerarşik Supervisor Ajanı (SupervisorAgent)
==============================================================

Çoklu Ajan Sistemi'nin (MAS) en üst düzey koordinatörüdür.
Worker ajanları belirli bir sırada (pipeline) çalıştırır, her
adımın sonucunu BlackBoard'dan okur ve hata durumunda Gemini LLM'e
danışarak recovery kararı verir.

Mimari Rolü:
    - Hierarchical MAS'ta "Manager/Supervisor" katmanını temsil eder.
    - Worker'ları doğrudan çağırır ancak iş mantığını delege eder.
    - Veri akışı BlackBoard üzerinden gerçekleşir; Supervisor yalnızca
      başarı/başarısızlık kontrol eder.

Pipeline Sırası:
    1. AuditWorker         → Girdi doğrulama
    2. HeadDetectionWorker → Kafa profili doğrulama (Gemini Vision)
    3. PreprocessingWorker → Tensör hazırlığı
    4. RecognitionWorker   → Embedding çıkarma
    5. EvaluationWorker    → Cosine similarity eşleşme
    6. ReportingWorker     → Gemini ile rapor üretimi

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Supervisor yalnızca koordinasyon yapar; iş mantığı
#        worker'larda yaşar.
# OCP  : Pipeline'a yeni bir ajan eklemek için PIPELINE listesine
#        ve workers dict'ine kayıt eklemek yeterlidir. Mevcut
#        worker'lar değişmez.
#        Not: Worker'ları otomatik keşfeden (plugin registry) bir yapı
#        daha güçlü OCP sağlardı; ancak ödev kapsamında açık liste
#        yaklaşımı tercih edilmiştir — pipeline sırası kritik olduğu
#        için bu bilinçli bir trade-off'tur.
# DIP  : Supervisor, BaseWorker arayüzüne bağımlıdır. Somut worker
#        sınıfları yalnızca __init__'te oluşturulur.
# Mediator Pattern: Supervisor, ajanlar arası koordinasyonun tek
#        noktasıdır; ajanlar birbirine referans vermez.
# ─────────────────────────────────────────────────────────────
"""

import os
from datetime import datetime

import google.generativeai as genai

from blackboard import BlackBoard
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, LOG_DIR
from agents.audit import AuditWorker
from agents.head_detection import HeadDetectionWorker
from agents.preprocessing import PreprocessingWorker
from agents.recognition import RecognitionWorker
from agents.evaluation import EvaluationWorker
from agents.reporting import ReportingWorker


class SupervisorAgent:
    """
    Tüm worker ajanları koordine eden Gemini LLM destekli Supervisor.

    Sorumlulukları:
        1. Görevi worker'lara sırayla delege eder.
        2. Her adımın sonucunu BlackBoard'dan kontrol eder.
        3. Hata durumunda Gemini'ye danışarak recovery kararı verir.
        4. Misyon logunu dosyaya kaydeder.

    Attributes:
        bb: Paylaşılan BlackBoard nesnesi.
        llm: Gemini GenerativeModel istemcisi (recovery kararları için).
        workers: Pipeline adı → Worker nesnesi eşlemesi.
    """

    # Worker çalıştırma sırası — pipeline sırası kritik!
    PIPELINE: list[str] = [
        "audit",
        "head_detection",
        "preprocessing",
        "recognition",
        "evaluation",
        "reporting",
    ]

    def __init__(self, image_path: str) -> None:
        """
        Supervisor'ı başlatır: BlackBoard oluşturur, Gemini'yi yapılandırır
        ve tüm worker'ları aynı BlackBoard ile ilklendirir.

        Args:
            image_path: Sorgulanacak görselin dosya yolu.
        """
        # Gemini API yapılandırması — process-wide tek seferlik.
        # Worker'lar bu çağrıya güvenir; kendi içlerinde tekrar yapmazlar.
        genai.configure(api_key=GEMINI_API_KEY)

        self.bb = BlackBoard()
        self.bb.query_image_path = image_path

        self.llm = genai.GenerativeModel(GEMINI_MODEL_NAME)

        # Worker'ları başlat (hepsi aynı BlackBoard'u paylaşır)
        self.workers = {
            "audit":          AuditWorker(self.bb),
            "head_detection": HeadDetectionWorker(self.bb),
            "preprocessing":  PreprocessingWorker(self.bb),
            "recognition":    RecognitionWorker(self.bb),
            "evaluation":     EvaluationWorker(self.bb),
            "reporting":      ReportingWorker(self.bb),
        }

    def delegate(self, worker_name: str) -> bool:
        """
        Belirtilen worker'a görev delege eder ve sonucu döndürür.

        Args:
            worker_name: PIPELINE listesindeki worker adı.

        Returns:
            True: Worker başarıyla tamamlandı.
            False: Worker başarısız oldu.
        """
        worker = self.workers[worker_name]
        self.bb.set_step(worker_name.upper())
        self.bb.log("Supervisor", f"Delegating → [{worker.name}]")

        success = worker.execute()

        if success:
            self.bb.log("Supervisor", f"[{worker.name}] completed ✅")
        else:
            self.bb.log("Supervisor", f"[{worker.name}] failed ❌")

        return success

    def run_mission(self) -> BlackBoard:
        """
        Tüm pipeline'ı yönetir.

        Her adımda BlackBoard'u kontrol eder. Hata durumunda Gemini'ye
        danışarak devam/dur kararı verir.

        Returns:
            BlackBoard: Görev sonucu — mission_status ve match_result
            alanlarından okunabilir.
        """
        self.bb.mission_status = "RUNNING"
        self.bb.log("Supervisor", "=== MİSYON BAŞLADI ===")

        for step in self.PIPELINE:
            success = self.delegate(step)

            if not success:
                # Gemini'ye danış: recovery mümkün mü?
                recovery = self._consult_gemini_for_recovery(step)

                if not recovery["continue"]:
                    self.bb.log("Supervisor", f"Durduruluyor: {recovery['reason']}")
                    self._save_mission_log()
                    return self.bb

                self.bb.log("Supervisor", f"Recovery: {recovery['reason']}")

        self.bb.mission_status = "SUCCESS"
        self.bb.log("Supervisor", "=== MİSYON TAMAMLANDI ===")
        self._save_mission_log()
        return self.bb

    def _consult_gemini_for_recovery(self, failed_step: str) -> dict:
        """
        Worker başarısız olunca Gemini'ye recovery danışması yapar.

        Args:
            failed_step: Başarısız olan pipeline adımı.

        Returns:
            {"continue": bool, "reason": str} formatında karar.
        """
        prompt = f"""
        Kaplumbağa kimlik tespit sisteminde '{failed_step}' adımı başarısız oldu.
        Hata: {self.bb.error_message}

        Bu hata kurtarılabilir mi?
        Sadece şu formatta yanıt ver:
        KARAR: DEVAM veya DUR
        NEDEN: (tek cümle Türkçe)
        """
        try:
            response = self.llm.generate_content(prompt)
            text = response.text
            continue_mission = "KARAR: DEVAM" in text
            reason = next(
                (line.replace("NEDEN:", "").strip()
                 for line in text.split("\n") if "NEDEN:" in line),
                "Gemini analizi tamamlandı."
            )
            return {"continue": continue_mission, "reason": reason}
        except Exception:
            # Gemini'ye ulaşamazsa güvenli tarafta kal
            return {"continue": False, "reason": "Gemini'ye ulaşılamadı, güvenli duruş."}

    def _save_mission_log(self) -> None:
        """BlackBoard'daki misyon logunu dosyaya yazar."""
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "mission_log.md"), "a", encoding="utf-8") as f:
            f.write(f"\n## Görev — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**Durum:** {self.bb.mission_status}\n\n")
            for entry in self.bb.mission_log:
                f.write(f"- {entry}\n")
            f.write("\n---\n")
