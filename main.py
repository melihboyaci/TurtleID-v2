"""
main.py — TurtleID Uygulama Giriş Noktası
===========================================

Deniz kaplumbağası kimlik tespit sisteminin ana giriş noktasıdır.
Kullanıcının ``data/query/`` klasörüne koyduğu kırpılmış kafa profili
görselini alır, SupervisorAgent üzerinden pipeline'ı çalıştırır ve
sonucu ekrana yazar.

Çalıştırma:
    python main.py

Kullanım Koşulları:
    1. ``data/query/`` klasöründe en az bir .jpg/.jpeg/.png görsel olmalı.
    2. ``.env`` dosyasında geçerli bir GEMINI_API_KEY tanımlanmış olmalı.
    3. ``turtle_embedding_model.keras`` dosyası kök dizinde bulunmalı.

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : main.py yalnızca programın giriş noktasıdır; iş mantığı
#        SupervisorAgent ve worker'larda yaşar.
# Facade Pattern: main() fonksiyonu, karmaşık MAS pipeline'ını
#        tek bir çağrıyla (supervisor.run_mission()) başlatır.
# ─────────────────────────────────────────────────────────────
"""

import os
import sys

from agents.supervisor import SupervisorAgent
from report_manager import ReportManager
from config import QUERY_DIR


def main() -> None:
    """
    Pipeline'ı başlatır ve sonucu ekrana yazdırır.

    Adımlar:
        1. Query klasöründeki ilk uygun görseli seçer.
        2. SupervisorAgent ile tüm pipeline'ı çalıştırır.
        3. ReportManager ile yapısal log kaydeder.
        4. Sonucu kullanıcıya gösterir.
    """
    files = [f for f in os.listdir(QUERY_DIR)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        print("❌ data/query/ klasörüne kırpılmış kafa profili görseli koy!")
        print("   Beklenen: net deniz kaplumbağası sağ/sol yan kafa profili.")
        sys.exit(1)

    query_path = os.path.join(QUERY_DIR, files[0])
    print(f"\n🐢 TurtleID — Kimlik Tespit Başlıyor")
    print(f"📸 Görsel: {files[0]} (kırpılmış kafa profili olarak doğrulanacak)\n")

    # Supervisor'ı başlat — tüm ajanları koordine eder
    supervisor = SupervisorAgent(image_path=query_path)
    blackboard = supervisor.run_mission()
    report_manager = ReportManager()
    report_manager.log_mission(blackboard)

    # Sonucu yazdır
    print("\n" + "=" * 40)
    print("📋 TESPİT SONUCU")
    print("=" * 40)

    if blackboard.mission_status == "SUCCESS":
        result = blackboard.match_result
        print(f"✅ Kimlik   : {result['name']}")
        print(f"📊 Benzerlik: %{result['score']*100:.1f}")
        print(f"🏷️  Durum    : {result['status']}")
    else:
        print(f"❌ Hata: {blackboard.error_message}")

    print("=" * 40)
    print(f"📝 Detaylı log: logs/mission_log.md")
    print(f"📈 Gelişim raporu: gelisim_raporu.md\n")


if __name__ == "__main__":
    main()
