import os
import google.generativeai as genai
from dotenv import load_dotenv
from blackboard import BlackBoard
from agents.audit import AuditWorker
from agents.head_detection import HeadDetectionWorker
from agents.preprocessing import PreprocessingWorker
from agents.recognition import RecognitionWorker
from agents.evaluation import EvaluationWorker
from agents.reporting import ReportingWorker


class SupervisorAgent:
    """
    Tüm worker ajanları koordine eden, Gemini LLM destekli
    Hiyerarşik Supervisor.

    Sorumlulukları:
    1. Görevi worker'lara delege eder
    2. Her adımın sonucunu blackboard'dan okur
    3. Hata durumunda recovery kararı verir
    4. Gemini ile nihai karar analizi yapar
    """

    # Worker çalıştırma sırası
    PIPELINE = [
        "audit",
        "head_detection",
        "preprocessing",
        "recognition",
        "evaluation",
        "reporting",
    ]

    def __init__(self, image_path: str):
        load_dotenv()
        self.bb = BlackBoard()
        self.bb.query_image_path = image_path

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel("gemini-1.5-flash")

        # Worker'ları başlat (hepsi aynı blackboard'u paylaşır)
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
        Supervisor'ın ana metodu: worker'a görevi delege eder,
        sonucu blackboard'dan okur, başarı/başarısızlık döner.
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
        Tüm pipeline'ı yönetir. Her adımda blackboard'u kontrol eder.
        Hata durumunda Gemini'ye danışarak recovery kararı verir.
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
        Worker başarısız olunca Gemini'ye sorar:
        'Devam etmeli miyiz yoksa durmalı mıyız?'
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
                (l.replace("NEDEN:", "").strip()
                 for l in text.split("\n") if "NEDEN:" in l),
                "Gemini analizi tamamlandı."
            )
            return {"continue": continue_mission, "reason": reason}
        except Exception:
            # Gemini'ye ulaşamazsa güvenli tarafta kal
            return {"continue": False, "reason": "Gemini'ye ulaşılamadı, güvenli duruş."}

    def _save_mission_log(self) -> None:
        """Blackboard'daki logu dosyaya yazar."""
        os.makedirs("logs", exist_ok=True)
        with open("logs/mission_log.md", "a", encoding="utf-8") as f:
            from datetime import datetime
            f.write(f"\n## Görev — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**Durum:** {self.bb.mission_status}\n\n")
            for entry in self.bb.mission_log:
                f.write(f"- {entry}\n")
            f.write("\n---\n")
