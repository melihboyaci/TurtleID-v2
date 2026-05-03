# 🐢 Kaplumbağa Veritabanı

Her kaplumbağa için klasör aç, içine `sag_profil.jpg`, `sol_profil.jpg` ve `metadata.json` koy.

## Klasör Yapısı

```
data/database/
├── ornek_kaplumbaga/
│   ├── sag_profil.jpg     ← sağ taraf yüz profili
│   ├── sol_profil.jpg     ← sol taraf yüz profili
│   └── metadata.json      ← kimlik bilgileri
└── ...
```

## metadata.json Formatı

```json
{
  "id": "turtle_001",
  "name": "Örnek Kaplumbağa",
  "species": "Chelonia mydas",
  "registered_at": "2026-05-02",
  "sighting_count": 1,
  "notes": "İlk kayıt"
}
```

## Bilimsel Temel

Chabrolle & Dumont-Dayot (2015) çalışmasına göre, her kaplumbağanın **sağ ve sol** post-oküler scut deseni asimetrik ve bireye özgüdür. Bu yüzden her birey için **iki profilin ortalaması** parmak izi vektörü olarak kullanılır.

> Sadece bir profil mevcutsa, sistem o tek profili kullanarak da çalışır — ancak en yüksek doğruluk için iki profilin de yüklenmesi önerilir.
