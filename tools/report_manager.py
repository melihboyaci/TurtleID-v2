"""
tools/report_manager.py — Yapısal Misyon Raporu Yöneticisi
=========================================================

Misyon sonucunu BlackBoard'dan okuyarak dosya tabanlı, yapısal
(structured) markdown rapora dönüştürür. Her çalıştırmada
``logs/mission_report.md`` dosyasına ekleme (append) yapar.

Bu modül ile ``ReportingWorker`` arasındaki fark:
    - ReportManager   : BlackBoard verilerini **olduğu gibi** yapısal
                        formatta kaydeder (ham veri kaydı).
    - ReportingWorker : Gemini LLM ile **analitik yorum** üretir
                        (AI destekli analiz).

İkisi birlikte kullanılarak hem ham veri hem de AI yorumu
saklanmış olur.

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Yalnızca BlackBoard → markdown dönüşümü ve dosya yazımı
#        yapar. AI çağrısı veya pipeline koordinasyonu yoktur.
# OCP  : Rapor formatı _build_report() metodu değiştirilerek
#        genişletilebilir; log_mission() değişmez.
# DIP  : BlackBoard'a tip bağımlılığı yoktur (Any ile loose coupling).
#        Bu sayede farklı veri yapılarıyla da çalışabilir.
# ─────────────────────────────────────────────────────────────
"""

import os
from datetime import datetime
from typing import Any

from config import LOG_DIR


class ReportManager:
    """
    Misyon sonucunu dosya tabanlı özet rapora dönüştürür.

    Attributes:
        REPORT_FILE: Rapor dosyasının adı.
    """

    REPORT_FILE: str = "mission_report.md"

    def log_mission(self, blackboard: Any) -> None:
        """
        BlackBoard içeriğini kalıcı görev raporuna yazar.

        Args:
            blackboard: Misyon verilerini içeren BlackBoard nesnesi.
        """
        os.makedirs(LOG_DIR, exist_ok=True)
        report_path = os.path.join(LOG_DIR, self.REPORT_FILE)
        with open(report_path, "a", encoding="utf-8") as file:
            file.write(self._build_report(blackboard))

    def _build_report(self, blackboard: Any) -> str:
        """
        BlackBoard verilerinden markdown rapor metni üretir.

        Args:
            blackboard: Misyon verilerini içeren BlackBoard nesnesi.

        Returns:
            Markdown formatında rapor metni.
        """
        result = getattr(blackboard, "match_result", {}) or {}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"## Görev Raporu — {timestamp}",
            "",
            f"**Durum:** {getattr(blackboard, 'mission_status', 'Bilinmiyor')}",
            f"**Sorgu Görseli:** {getattr(blackboard, 'query_image_path', '')}",
            f"**Kimlik:** {result.get('name', 'Bilinmiyor')}",
            f"**Benzerlik:** %{result.get('score', 0) * 100:.1f}",
            f"**Eşleşme Durumu:** {result.get('status', 'Bilinmiyor')}",
            f"**Profil Notu:** {result.get('profile_note', 'Yok')}",
            "",
            "### Misyon Logu",
            "",
        ]
        lines.extend(f"- {entry}" for entry in getattr(blackboard, "mission_log", []))
        lines.extend(["", "---", ""])
        return "\n".join(lines)
