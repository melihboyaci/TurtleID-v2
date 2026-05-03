import os
import sys
from agents.supervisor import SupervisorAgent


def main() -> None:
    # Query klasöründeki ilk görseli al
    query_dir = "data/query"
    files = [f for f in os.listdir(query_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        print("❌ data/query/ klasörüne tanımlanacak görsel koy!")
        sys.exit(1)

    query_path = os.path.join(query_dir, files[0])
    print(f"\n🐢 TurtleID — Kimlik Tespit Başlıyor")
    print(f"📸 Görsel: {files[0]}\n")

    # Supervisor'ı başlat — tüm ajanları koordine eder
    supervisor = SupervisorAgent(image_path=query_path)
    blackboard = supervisor.run_mission()

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
