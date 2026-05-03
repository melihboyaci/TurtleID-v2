import os
from datetime import datetime
from typing import Any


class ReportManager:
    """Misyon sonucunu dosya tabanlı özet rapora dönüştürür."""

    REPORT_DIR = "logs"
    REPORT_FILE = "mission_report.md"

    def log_mission(self, blackboard: Any) -> None:
        """Blackboard içeriğini kalıcı görev raporuna yazar."""
        os.makedirs(self.REPORT_DIR, exist_ok=True)
        report_path = os.path.join(self.REPORT_DIR, self.REPORT_FILE)
        with open(report_path, "a", encoding="utf-8") as file:
            file.write(self._build_report(blackboard))

    def _build_report(self, blackboard: Any) -> str:
        """Blackboard verilerinden markdown rapor metni üretir."""
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
