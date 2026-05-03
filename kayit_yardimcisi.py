import os
import json
import shutil
from datetime import datetime
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

DATABASE_DIR = "data/database"

SPECIES_MAP = {
    "yeşil": "Chelonia mydas",
    "green": "Chelonia mydas",
    "hawksbill": "Eretmochelys imbricata",
    "camız": "Eretmochelys imbricata",
    "loggerhead": "Caretta caretta",
    "caretta": "Caretta caretta",
    "leatherback": "Dermochelys coriacea",
    "deri": "Dermochelys coriacea",
}


def detect_species(image_path: str) -> dict:
    """
    Gemini Vision ile görselden tür tespiti yapar.
    Döner: {species_latin, species_turkish, profile_side, confidence}
    """
    with Image.open(image_path) as pil_img:
        pil_img_copy = pil_img.copy()  # Belleğe kopyala, dosya handle'ı kapanacak

    prompt = """Bu deniz kaplumbağası görselini incele.

Sadece şu formatta yanıt ver:
TÜR_LATINCE: (Chelonia mydas / Eretmochelys imbricata / Caretta caretta / Dermochelys coriacea / Bilinmiyor)
TÜR_TÜRKÇE: (Yeşil Kaplumbağa / Hawksbill / Caretta Caretta / Deri Sırtlı / Bilinmiyor)
PROFİL: (sag / sol / onde / arkada / belirsiz)
GÜVEN: (yüksek / orta / düşük)
NOT: (tek cümle gözlem)"""

    try:
        response = model.generate_content([prompt, pil_img_copy])
        text = response.text.strip()
        result = {}
        for line in text.split("\n"):
            if "TÜR_LATINCE:" in line:
                result["species_latin"] = line.replace("TÜR_LATINCE:", "").strip()
            elif "TÜR_TÜRKÇE:" in line:
                result["species_turkish"] = line.replace("TÜR_TÜRKÇE:", "").strip()
            elif "PROFİL:" in line:
                result["profile_side"] = line.replace("PROFİL:", "").strip()
            elif "GÜVEN:" in line:
                result["confidence"] = line.replace("GÜVEN:", "").strip()
            elif "NOT:" in line:
                result["note"] = line.replace("NOT:", "").strip()
        return result
    except Exception as e:
        print(f"⚠️  Gemini hatası: {e}")
        print("\n📝 Manuel giriş yapabilirsin (veya boş bırak 'Bilinmiyor' olarak kaydeder):")
        species_latin = input("   Tür (latince) [Caretta caretta]: ").strip()
        if not species_latin:
            species_latin = "Caretta caretta"
        species_turkish = input("   Tür (Türkçe) [Caretta Caretta]: ").strip()
        if not species_turkish:
            species_turkish = "Caretta Caretta"
        profile_side = input("   Profil [sag/sol/belirsiz]: ").strip()
        if not profile_side:
            profile_side = "belirsiz"
        note = input("   Not: ").strip()
        return {
            "species_latin": species_latin,
            "species_turkish": species_turkish,
            "profile_side": profile_side,
            "confidence": "manuel",
            "note": note or "Manuel giriş yapıldı."
        }


def register_turtle(
    image_path: str,
    name: str,
    profile: str,          # "sag" veya "sol"
    notes: str = ""
) -> bool:
    """
    Tek bir görseli veritabanına kaydeder.
    - Gemini ile tür tespiti yapar
    - Klasör oluşturur
    - Görseli sag_profil.jpg veya sol_profil.jpg olarak kopyalar
    - metadata.json yazar
    """
    print(f"\n📸 Görsel analiz ediliyor: {os.path.basename(image_path)}")
    
    # Gemini ile tür tespiti
    species_info = detect_species(image_path)
    print(f"🔍 Gemini tespiti:")
    print(f"   Tür: {species_info.get('species_latin')} ({species_info.get('species_turkish')})")
    print(f"   Profil: {species_info.get('profile_side')}")
    print(f"   Güven: {species_info.get('confidence')}")
    print(f"   Not: {species_info.get('note')}")

    # Klasör oluştur
    folder_name = name.lower().replace(" ", "_")
    folder_path = os.path.join(DATABASE_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Görseli kopyala
    target_filename = f"{profile}_profil.jpg"
    target_path = os.path.join(folder_path, target_filename)
    shutil.copy2(image_path, target_path)
    print(f"✅ Görsel kaydedildi: {target_path}")

    # metadata.json oluştur veya güncelle
    meta_path = os.path.join(folder_path, "metadata.json")
    
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["sighting_count"] = metadata.get("sighting_count", 1) + 1
        print(f"📝 Metadata güncellendi (mevcut kayıt)")
    else:
        turtle_id = f"turtle_{len(os.listdir(DATABASE_DIR)):03d}"
        metadata = {
            "id": turtle_id,
            "name": name,
            "species": species_info.get("species_latin", "Bilinmiyor"),
            "species_turkish": species_info.get("species_turkish", "Bilinmiyor"),
            "registered_at": datetime.now().strftime("%Y-%m-%d"),
            "sighting_count": 1,
            "gemini_confidence": species_info.get("confidence", "düşük"),
            "notes": notes or species_info.get("note", "")
        }
        print(f"📝 Yeni metadata oluşturuldu")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"🐢 '{name}' başarıyla kaydedildi!\n")
    return True


def interactive_register():
    """
    Kullanıcıyla interaktif kayıt yapar.
    Her görsel için isim ve hangi profil olduğunu sorar.
    """
    print("\n" + "="*50)
    print("🐢 TurtleID — Kaplumbağa Kayıt Yardımcısı")
    print("="*50)
    print("Tür tespiti Gemini Vision tarafından yapılır.")
    print("Çıkmak için 'q' yaz.\n")

    while True:
        # Görsel yolu
        image_path = input("📁 Görsel yolu (veya 'q'): ").strip()
        if image_path.lower() == 'q':
            break
        if not os.path.exists(image_path):
            print("❌ Dosya bulunamadı, tekrar dene.")
            continue

        # Kaplumbağa adı
        name = input("🏷️  Bu kaplumbağaya bir isim ver: ").strip()
        if not name:
            print("❌ İsim boş olamaz.")
            continue

        # Profil tarafı
        print("📐 Bu hangi profil?")
        print("   1 → Sağ profil")
        print("   2 → Sol profil")
        choice = input("Seçim (1/2): ").strip()
        profile = "sag" if choice == "1" else "sol"

        # Ek not (opsiyonel)
        notes = input("📝 Ek not (opsiyonel, boş bırakabilirsin): ").strip()

        # Kaydet
        register_turtle(image_path, name, profile, notes)

        # Devam?
        cont = input("Başka görsel eklemek ister misin? (e/h): ").strip().lower()
        if cont != 'e':
            break

    print("\n✅ Kayıt işlemi tamamlandı!")
    print(f"📂 Veritabanı: {DATABASE_DIR}/")


if __name__ == "__main__":
    interactive_register()
