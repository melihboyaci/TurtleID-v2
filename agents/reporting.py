"""
reporting.py — Rapor Üretim Ajanı (ReportingWorker)
=====================================================

Pipeline'ın son adımı. Gemini LLM ile misyon sonucunu analiz eder
ve insana yönelik gelişim raporu (gelisim_raporu.md) üretir.

BlackBoard'daki tüm sonuç verisini okur, Gemini'ye gönderir ve
dönen doğal dil raporunu dosyaya yazar.

Bu worker ile ``ReportManager`` arasındaki fark:
    - ReportingWorker : Gemini ile **analitik** rapor üretir (AI yorumu).
    - ReportManager   : BlackBoard verilerini **yapısal** olarak kaydeder (ham log).

BlackBoard Akışı:
    Okur  : match_result, mission_log
    Yazar : — (dosyaya yazar, BlackBoard'a yeni veri yazmaz)

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Yalnızca Gemini ile rapor üretir ve dosyaya yazar.
#        Eşleşme puanlama veya embedding üretimi bu worker'ın
#        sorumluluğu dışındadır.
# OCP  : Prompt değiştirilerek farklı rapor formatları üretilebilir;
#        mevcut yapı değişmez.
# DIP  : Gemini model adı, API anahtarı ve rapor dosya yolu
#        config.py'den alınır.
# Fail-Safe: Raporlama hatası misyonu durdurmaz (return True).
# ─────────────────────────────────────────────────────────────
"""

import os
from datetime import datetime

import google.generativeai as genai

from agents import BaseWorker
from config import GEMINI_MODEL_NAME, REPORT_FILE


class ReportingWorker(BaseWorker):
    """
    Gemini LLM ile misyon sonucunu analiz ederek gelişim raporu yazar.

    Attributes:
        llm: Gemini GenerativeModel istemcisi.
    """

    def __init__(self, blackboard) -> None:
        super().__init__(blackboard)
        self.llm = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def execute(self) -> bool:
        """
        Gemini'ye prompt gönderir, dönen raporu dosyaya yazar.

        Returns:
            True: Her durumda (raporlama hatası misyonu durdurmamalı).
        """
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
        """
        Rapor içeriğini tarih damgasıyla birlikte dosyaya ekler.

        Args:
            content: Gemini tarafından üretilen rapor metni.
        """
        if not os.path.exists(REPORT_FILE):
            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                f.write("# 🐢 Kaplumbağa Tanıma — Gelişim Raporu\n\n")

        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(f"## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(content + "\n\n---\n\n")
