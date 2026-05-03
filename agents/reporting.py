import os
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from agents import BaseWorker


class ReportingWorker(BaseWorker):
    """
    Gemini LLM ile mission sonucunu analiz eder ve rapor yazar.
    Blackboard'daki tüm veriyi okur, gelisim_raporu.md'ye yazar.
    """

    REPORT_FILE = "gelisim_raporu.md"

    def __init__(self, blackboard):
        super().__init__(blackboard)
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel("gemini-1.5-flash")

    def execute(self) -> bool:
        result = self.bb.match_result
        log_summary = "\n".join(self.bb.mission_log[-10:])

        prompt = f"""
        Sen kıdemli bir yapay zeka araştırmacısısın.
        Kaplumbağa kimlik tespit sistemi şu sonucu üretti:

        Tespit: {result.get('name', 'Bilinmiyor')}
        Benzerlik: %{result.get('score', 0)*100:.1f}
        Durum: {result.get('status', 'HATA')}
        Sistem Logu: {log_summary}

        Şu formatta günlük rapor yaz:
        **Ne yapıldı:** (pipeline adımlarını özetle)
        **Sonuç:** (tespit sonucunu yorumla)
        **Problemler:** (düşük skor veya hataları açıkla)
        **İyileştirme:** (gelecek adımlar)
        """

        try:
            response = self.llm.generate_content(prompt)
            self._write_report(response.text)
            self.log("Rapor başarıyla yazıldı.")
            return True
        except Exception as e:
            self.log(f"Rapor yazılamadı: {e}")
            return True  # Raporlama hatası misyonu durdurmamalı

    def _write_report(self, content: str) -> None:
        if not os.path.exists(self.REPORT_FILE):
            with open(self.REPORT_FILE, "w", encoding="utf-8") as f:
                f.write("# 🐢 Kaplumbağa Tanıma — Gelişim Raporu\n\n")

        with open(self.REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(f"## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(content + "\n\n---\n\n")
