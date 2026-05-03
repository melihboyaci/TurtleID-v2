# 🐢 TurtleID v2

Deniz kaplumbağalarını **post-oküler scut** (göz çevresi pul) desenleri üzerinden tanıyan, **gerçek hiyerarşik çoklu ajan** mimarisiyle çalışan kimlik tespit sistemi.

## Mimari

- **Pattern:** Hierarchical (Supervisor → Workers)
- **İletişim:** Shared Blackboard
- **LLM:** Google Gemini 1.5 Flash (supervisor recovery + vision verification + reporting)
- **Embedding:** ResNet50 (ImageNet)
- **Eşleştirme:** Cosine Similarity

```
SupervisorAgent
   ├── AuditWorker            (girdi doğrulama)
   ├── HeadDetectionWorker    (OpenCV + Gemini Vision)
   ├── PreprocessingWorker    (224x224 normalize)
   ├── RecognitionWorker      (ResNet50 embedding)
   ├── EvaluationWorker       (Cosine Similarity)
   └── ReportingWorker        (Gemini LLM raporu)
```

Tüm worker'lar tek bir `BlackBoard` üzerinden okur/yazar — birbirlerini doğrudan çağırmazlar. Sadece `SupervisorAgent` worker'ları `delegate()` ile çağırır.

## Kurulum

```bash
pip install -r gereksinimler.txt
cp .env.example .env
# .env dosyasına GEMINI_API_KEY değerini yaz
```

## Klasör Yapısı

```
turtle-id/
├── main.py
├── blackboard.py
├── agents/
│   ├── __init__.py          # BaseWorker
│   ├── supervisor.py
│   ├── audit.py
│   ├── head_detection.py
│   ├── preprocessing.py
│   ├── recognition.py
│   ├── evaluation.py
│   └── reporting.py
├── data/
│   ├── database/            # Her kaplumbağa için ayrı klasör
│   │   └── ornek_kaplumbaga/
│   │       ├── sag_profil.jpg
│   │       ├── sol_profil.jpg
│   │       └── metadata.json
│   └── query/               # Tanımlanacak görsel
├── logs/
│   └── mission_log.md
├── .env
├── gereksinimler.txt
└── README.md
```

## Kullanım

1. `data/database/` altında her kaplumbağa için ayrı klasör aç.
2. Her klasöre `sag_profil.jpg`, `sol_profil.jpg` ve `metadata.json` koy.
3. `data/query/` içine tanımlanacak görseli koy.
4. Çalıştır:

```bash
python main.py
```

## Çıktılar

- `logs/mission_log.md` — Her görevin detaylı adım logu
- `gelisim_raporu.md` — Gemini tarafından yazılmış günlük gelişim raporu

## Eşleşme Eşikleri

| Skor   | Durum         |
| ------ | ------------- |
| ≥ 0.85 | GÜÇLÜ_EŞLEŞME |
| ≥ 0.70 | OLASI_EŞLEŞME |
| < 0.70 | YENİ_BİREY    |
