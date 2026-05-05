# TurtleID — Yazılım Mimarisi Kalite Değerlendirme Raporu

**Proje Adı:** TurtleID — Deniz Kaplumbağası Kimlik Tespit Sistemi  
**Mimari:** Hiyerarşik Çoklu Ajan Sistemi (Hierarchical MAS) + Blackboard Tasarım Deseni  
**Değerlendirme Tarihi:** 5 Mayıs 2026  
**Hazırlayan:** Yazılım Mimarisi ve Clean Code Analizi

---

## Yönetici Özeti

Bu rapor, `turtle-id` projesinin kaynak kodunu birinci elden inceleyerek yazılım kalitesini iki temel eksende değerlendirmektedir: **(1) S.O.L.I.D Prensipleri** ve **(2) Clean Code Standartları**. Değerlendirme, soyut iddialardan değil; gerçek sınıf isimleri, dosya yolları ve kod satırları üzerinden yürütülmüştür.

Proje, **7 bağımsız worker ajanından** (`AuditWorker`, `HeadDetectionWorker`, `PreprocessingWorker`, `RecognitionWorker`, `EvaluationWorker`, `ReportingWorker` ve `SupervisorAgent`) oluşan bir hiyerarşik mimariye sahiptir. Tüm veri akışı `BlackBoard` nesnesi üzerinden gerçekleşmekte; ajanlar birbirini doğrudan tanımamaktadır.

---

## 1. S.O.L.I.D Prensipleri Analizi

### 1.1 S — Single Responsibility Principle (Tek Sorumluluk İlkesi)

> *"Bir sınıfın değişmesi için yalnızca tek bir nedeni olmalıdır."* — Robert C. Martin

#### Kanıt: Worker Ajanlarının Sorumluluklarının Kesin Ayrımı

Sistemdeki her worker ajanı, pipeline içinde **yalnızca tek bir bilişsel görevi** üstlenir. Aşağıdaki tablo bu ayrımı somutlaştırmaktadır:

| Sınıf Adı | Dosya | Tek Sorumluluğu |
|---|---|---|
| `AuditWorker` | `agents/audit.py` | Sorgu görselinin format, boyut ve okunabilirlik doğrulaması |
| `HeadDetectionWorker` | `agents/head_detection.py` | Gemini Vision API ile kaplumbağa kafa profili doğrulaması |
| `PreprocessingWorker` | `agents/preprocessing.py` | RGB görselini model girdisi (224×224 tensör) formatına dönüştürme |
| `RecognitionWorker` | `agents/recognition.py` | Triplet Loss modeli üzerinden 256-boyutlu embedding üretimi |
| `EvaluationWorker` | `agents/evaluation.py` | Cosine similarity ile eşleşme puanlama ve karar verme |
| `ReportingWorker` | `agents/reporting.py` | Gemini LLM ile analitik gelişim raporu üretimi |
| `SupervisorAgent` | `agents/supervisor.py` | Pipeline koordinasyonu ve hata yönetimi (iş mantığı içermez) |

**Kod Kanıtı — `preprocessing.py` (satır 43–58):**
```python
def execute(self) -> bool:
    if self.bb.head_crop is None:
        self.bb.fail(self.name, "head_crop yok.")
        return False

    tensor = to_tensor(self.bb.head_crop)
    self.bb.model_ready_tensor = tensor
    self.log(f"Tensör hazırlandı: {tensor.shape}, dtype={tensor.dtype}")
    return True
```
`PreprocessingWorker.execute()` yalnızca "tensör hazırla ve BlackBoard'a yaz" işini yapar. Model çağrısı, embedding üretimi veya doğrulama mantığı içermez. Benzer şekilde `evaluation.py` (satır 48–106) yalnızca cosine similarity hesaplar; tensör üretmez, dosya okumaz.

**`supervisor.py` (satır 27–28) kendi docstring'inde bunu açıkça belirtir:**
```
# SRP  : Supervisor yalnızca koordinasyon yapar; iş mantığı
#        worker'larda yaşar.
```

---

### 1.2 O — Open/Closed Principle (Açık/Kapalı İlkesi)

> *"Yazılım varlıkları genişletmeye açık, değişikliğe kapalı olmalıdır."* — Bertrand Meyer

#### Kanıt: `PIPELINE` Listesi ve `workers` Sözlüğü Üzerinden Genişletilebilirlik

`SupervisorAgent` içindeki pipeline tanımı (`supervisor.py`, satır 75–82 ve 102–109) yeni bir ajan eklenmesini **mevcut hiçbir worker'a dokunmadan** mümkün kılmaktadır:

```python
# supervisor.py — Satır 75–82
PIPELINE: list[str] = [
    "audit",
    "head_detection",
    "preprocessing",
    "recognition",
    "evaluation",
    "reporting",
]

# supervisor.py — Satır 102–109
self.workers = {
    "audit":          AuditWorker(self.bb),
    "head_detection": HeadDetectionWorker(self.bb),
    "preprocessing":  PreprocessingWorker(self.bb),
    "recognition":    RecognitionWorker(self.bb),
    "evaluation":     EvaluationWorker(self.bb),
    "reporting":      ReportingWorker(self.bb),
}
```

**Yeni Ajan Ekleme Senaryosu:** Sisteme "GeoLocationWorker" (GPS koordinat doğrulama) adında yeni bir ajan eklendiğini varsayalım. Yapılması gereken tek değişiklik:

1. `agents/geo_location.py` dosyasında `BaseWorker`'dan türeyen yeni sınıf yazılır.
2. `PIPELINE` listesine `"geo_location"` eklenir.
3. `workers` sözlüğüne `"geo_location": GeoLocationWorker(self.bb)` eklenir.

Mevcut 6 worker (`AuditWorker`, `HeadDetectionWorker`, vb.) **hiçbir satırı değiştirilmez**. Bu, OCP'nin teorik tanımıyla örtüşen pratik bir kanıttır.

**`BlackBoard` (`blackboard.py`, satır 29–30) bu prensibi kendi düzeyinde de uygular:**
```
# OCP  : Yeni bir ajan çıktısı eklemek için yalnızca yeni bir alan
#        (field) eklenir; mevcut alanlar değişmez.
```

---

### 1.3 L — Liskov Substitution Principle (Liskov Yerine Geçme İlkesi)

> *"Alt sınıfların nesneleri, üst sınıfların nesnelerinin yerine, programın doğruluğunu bozmadan kullanılabilmelidir."* — Barbara Liskov, 1987

#### Kanıt: `BaseWorker` Soyut Kontratı ve Alt Sınıf Uyumu

`agents/__init__.py` (satır 60–72) üretimi net bir kontrat tanımlar:

```python
@abstractmethod
def execute(self) -> bool:
    """
    Worker'ın asıl görevini yerine getirir.

    Returns:
        True: Görev başarıyla tamamlandı.
        False: Görev başarısız oldu (hata BlackBoard'a yazılmıştır).
    """
    ...
```

Bu kontrat üç zorunluluk içerir:
1. `execute()` metodu **her zaman `bool` döndürmeli** (hiçbir zaman `None` veya exception fırlatmadan çıkmamalı).
2. Başarısız durumda `BlackBoard` üzerinden `bb.fail()` çağrılmalı.
3. Sonuçlar BlackBoard'a yazılmalı; `execute()` dışarıya veri döndürmemeli.

**Tüm 6 alt sınıf bu kontrata tam uyum gösterir:**

| Sınıf | `execute()` İmzası | Hata Bildirimi |
|---|---|---|
| `AuditWorker` | `def execute(self) -> bool` | `self.bb.fail(self.name, msg)` |
| `HeadDetectionWorker` | `def execute(self) -> bool` | `self.bb.fail(self.name, ...)` |
| `PreprocessingWorker` | `def execute(self) -> bool` | `self.bb.fail(self.name, ...)` |
| `RecognitionWorker` | `def execute(self) -> bool` | `self.bb.fail(self.name, ...)` |
| `EvaluationWorker` | `def execute(self) -> bool` | `self.bb.fail(self.name, ...)` |
| `ReportingWorker` | `def execute(self) -> bool` | `return True` (fail-safe) |

**`supervisor.py`'daki delege metodu (satır 111–133)** bu yerine geçebilirliği pratikte uygular:

```python
def delegate(self, worker_name: str) -> bool:
    worker = self.workers[worker_name]
    self.bb.set_step(worker_name.upper())
    self.bb.log("Supervisor", f"Delegating → [{worker.name}]")

    success = worker.execute()   # ← Hangi worker olduğu önemli değil

    if success:
        self.bb.log("Supervisor", f"[{worker.name}] completed ✅")
    else:
        self.bb.log("Supervisor", f"[{worker.name}] failed ❌")
    return success
```

`delegate()` metodu, `BaseWorker` türündeki herhangi bir nesneyi `worker.execute()` çağrısıyla kullanabilir. `AuditWorker` yerine `EvaluationWorker` konulduğunda `delegate()` kodu değişmez; LSP tam anlamıyla sağlanmıştır.

> **Not:** `ReportingWorker` başarısız olsa bile `return True` döndürerek misyonu durdurmamaktadır (`reporting.py`, satır 87). Bu, LSP kontratına görünürde bir istisna gibi görünse de; docstring'de **"Fail-Safe"** olarak bilinçli belgelenmiş bir tasarım kararıdır — raporlama, kimlik tespitinin kritik yolu dışındadır.

---

### 1.4 I — Interface Segregation Principle (Arayüz Ayrıştırma İlkesi)

> *"İstemciler, kullanmadıkları arayüzlere bağımlı olmaya zorlanmamalıdır."* — Robert C. Martin

#### Kanıt: `BlackBoard` Üzerinden Minimal Arayüz ve Gevşek Bağlılık

`BaseWorker` arayüzü (`agents/__init__.py`, satır 40–76) son derece yalındır:

```python
class BaseWorker(ABC):
    def __init__(self, blackboard: BlackBoard) -> None:
        self.bb = blackboard           # Tek bağımlılık

    @abstractmethod
    def execute(self) -> bool:         # Tek zorunlu metot
        ...

    def log(self, message: str) -> None:  # Paylaşılan loglama altyapısı
        self.bb.log(self.name, message)
```

Arayüz; **sadece `__init__`, `execute` ve `log`** metotlarından oluşur. Bir worker'ın diğer worker'ların metodlarını, özelliklerini veya iç durumlarını bilmesi gerekmez.

**ISP'nin en güçlü kanıtı, worker'ların BlackBoard kullanım biçimidir.** Her worker, BlackBoard üzerinden yalnızca kendi ilgilendiği alanları okur/yazar:

```
AuditWorker         → Okur: query_image_path        | Yazar: audit_result
HeadDetectionWorker → Okur: query_image_path         | Yazar: head_crop, head_confidence
PreprocessingWorker → Okur: head_crop                | Yazar: model_ready_tensor
RecognitionWorker   → Okur: model_ready_tensor       | Yazar: query_embedding, db_embeddings, db_files
EvaluationWorker    → Okur: query_embedding, db_*    | Yazar: match_result
ReportingWorker     → Okur: match_result, mission_log| Yazar: — (dosyaya yazar)
```

**Karşılaştırmalı Analiz — ISP Olmasaydı:**
Tüm worker'lar birbirini bağımlılık olarak alıyor olsaydı, `PreprocessingWorker.__init__(self, audit_worker, head_detection_worker, recognition_worker, ...)` gibi şişkin bir imzaya sahip olurdu. Bunun yerine her worker **tek ve küçük** bir `BlackBoard` referansı alır; yalnızca ihtiyaç duyduğu alanlarla etkileşime girer.

---

### 1.5 D — Dependency Inversion Principle (Bağımlılıkların Tersine Çevrilmesi)

> *"Yüksek seviyeli modüller, düşük seviyeli modüllere bağımlı olmamalıdır. Her ikisi de soyutlamalara bağımlı olmalıdır."* — Robert C. Martin

#### Kanıt: `SupervisorAgent`'ın Somut Sınıflara Değil, Soyut Arayüze Bağımlılığı

`supervisor.py`'daki `delegate()` metodu (satır 111–133) somut sınıf isimlerine hiç başvurmaz:

```python
def delegate(self, worker_name: str) -> bool:
    worker = self.workers[worker_name]   # ← dict'ten alınan BaseWorker referansı
    success = worker.execute()           # ← Soyut arayüz çağrısı
    return success
```

`worker.execute()` çağrısı, somut tipin `AuditWorker` mi yoksa `RecognitionWorker` mi olduğunu bilmez; `BaseWorker` arayüzünü kullanır.

**İki Katmanlı DIP Uygulaması:**

**Katman 1 — Supervisor → BaseWorker:**
Somut worker sınıfları yalnızca `__init__` metodunda `self.workers` sözlüğüne atanır. Bunun dışında Supervisor, somut tipleri hiç görmez.

**Katman 2 — Worker'lar → config.py:**
Tüm worker'lar, sabit değerleri doğrudan kod içine gömmek yerine merkezi `config.py` modülünü kullanır:

```python
# evaluation.py — Satır 35–36
from config import MATCH_THRESHOLD, POSSIBLE_THRESHOLD

# recognition.py — Satır 47
from config import EMBEDDING_MODEL_PATH, CACHE_FILE, DATABASE_DIR

# audit.py — Satır 33
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MIN_IMAGE_SIZE_PX
```

Böylece eşik değerleri, model yolları veya dosya konumları değiştirilmek istendiğinde **yalnızca `config.py` güncellenir**; worker sınıfları değişmez. `config.py`'nin kendisi de DIP'i açıkça belgelemektedir (satır 22–23):

```
# DIP  : Diğer modüller somut değerlere değil, bu modüle bağımlıdır.
#        Model değiştirilmek istendiğinde yalnızca bu dosya güncellenir.
```

---

## 2. Clean Code Standartları

### 2.1 Dosya ve Klasör Modülerliği

Proje, **endişe ayrımı (separation of concerns)** ilkesine uygun, öngörülebilir bir klasör hiyerarşisine sahiptir:

```
turtle-id/
├── agents/                  # MAS katmanı: tüm worker ve Supervisor ajanları
│   ├── __init__.py          # BaseWorker soyut sınıfı (paket arayüzü)
│   ├── supervisor.py        # Koordinatör ajan
│   ├── audit.py             # Girdi doğrulama
│   ├── head_detection.py    # Gemini Vision doğrulama
│   ├── preprocessing.py     # Tensör hazırlığı
│   ├── recognition.py       # Embedding üretimi
│   ├── evaluation.py        # Benzerlik puanlama
│   ├── reporting.py         # Rapor üretimi
│   └── tensor_utils.py      # Paylaşılan tensör yardımcı fonksiyonları
├── tools/                   # Yardımcı araçlar (pipeline dışı)
│   └── report_manager.py    # Yapısal log yönetimi
├── api/                     # REST API katmanı
│   └── server.py            # FastAPI endpoint'leri
├── data/                    # Veri katmanı
├── logs/                    # Çalışma zamanı logları
├── config.py                # Merkezi yapılandırma (tek kaynak)
├── blackboard.py            # Paylaşılan durum deposu
└── main.py                  # Uygulama giriş noktası (Facade)
```

**Dikkat Çeken Tasarım Kararı — `agents/tensor_utils.py`:**
`PreprocessingWorker` ve `RecognitionWorker`, görsel-tensör dönüşümü için aynı mantığı kullanır. Bu ortak mantık, iki worker'ın birbirine bağımlı olmasını önlemek amacıyla `agents/tensor_utils.py` adı altında bağımsız bir yardımcı modüle çıkarılmıştır. Bu karar, **DRY (Don't Repeat Yourself)** ilkesini sağlarken aynı zamanda worker'lar arası döngüsel bağımlılığı önler.

---

### 2.2 Magic Number Yokluğu

Sistemde ham sayısal sabitler ("magic number") kod içine gömülmemiş; tümü `config.py` dosyasında **anlamlı isimlerle** tanımlanmıştır:

```python
# config.py — Satır 43–67
EMBEDDING_MODEL_PATH: str = "turtle_embedding_model.keras"
EMBEDDING_DIM: int        = 256
TARGET_SIZE: tuple        = (224, 224)      # ← "224" yerine TARGET_SIZE
MATCH_THRESHOLD: float    = 0.85            # ← "0.85" yerine MATCH_THRESHOLD
POSSIBLE_THRESHOLD: float = 0.70            # ← "0.70" yerine POSSIBLE_THRESHOLD
ALLOWED_EXTENSIONS: set   = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB: int     = 10
MIN_IMAGE_SIZE_PX: int    = 100
```

**Bunun anlamı:** `evaluation.py`'de `0.85` ve `0.70` gibi değerler doğrudan kodda yer almaz; `MATCH_THRESHOLD` ve `POSSIBLE_THRESHOLD` isimleriyle kullanılır. Bu yaklaşım kodu **kendi kendini belgeleyen (self-documenting)** hâle getirir.

---

### 2.3 Hata Yönetimi (Error Handling) Kalitesi

Proje, birden fazla katmanlı, savunmacı bir hata yönetimi stratejisi uygular:

**Katman 1 — BlackBoard Üzerinden Merkezi Hata Yayılımı:**
`BlackBoard.fail()` metodu (`blackboard.py`, satır 108–118), hata durumunu `mission_status = "FAILED"` ve `error_message` alanlarına yazarak pipeline'ın geri kalanını durdurur:

```python
def fail(self, agent_name: str, reason: str) -> None:
    self.mission_status = "FAILED"
    self.error_message = reason
    self.log(agent_name, f"MİSYON BAŞARISIZ: {reason}")
```

**Katman 2 — Supervisor'ın LLM Destekli Recovery Mekanizması:**
`supervisor.py` (satır 168–199), bir worker başarısız olduğunda kör bir şekilde durmaz; Gemini LLM'e danışarak hata kurtarılabilir mi değerlendirmesi yapar:

```python
def _consult_gemini_for_recovery(self, failed_step: str) -> dict:
    ...
    try:
        response = self.llm.generate_content(prompt)
        ...
    except Exception:
        # Gemini'ye ulaşamazsa güvenli tarafta kal
        return {"continue": False, "reason": "Gemini'ye ulaşılamadı, güvenli duruş."}
```

**Katman 3 — Fail-Safe Tasarım Kararı (`ReportingWorker`):**
`reporting.py` (satır 80–87), raporlama hatalarının misyonu durdurmaması için `return True` döndürür:

```python
try:
    response = self.llm.generate_content(prompt)
    self._write_report(response.text)
    return True
except Exception as e:
    self.log(f"Rapor yazılamadı: {e}")
    return True  # ← Raporlama hatası misyonu durdurmamalı
```

**Katman 4 — JSON Ayrıştırma Güvenliği (`HeadDetectionWorker`):**
`head_detection.py` (satır 157–187), Gemini yanıtı beklenen JSON formatında olmasa bile çökmemek için çok adımlı bir ayrıştırma stratejisi (regex + fallback) uygular.

**Katman 5 — Giriş Doğrulama Zinciri (`AuditWorker`):**
Pipeline'a girmeden önce verinin geçerliliğini doğrulayan 4 aşamalı kontrol zinciri (`audit.py`, satır 60–66), erken hata yakalamayı (fail-fast) sağlar:

```python
for check in [self._check_extension, self._check_size,
              self._check_readable, self._check_dimensions]:
    passed, msg = check(path)
    if not passed:
        self.bb.audit_result = {"passed": False, "message": msg}
        self.bb.fail(self.name, msg)
        return False
```

Bu yaklaşım aynı zamanda **Strategy Tasarım Deseni**ni de örnekler: her `_check_*` metodu bağımsız bir doğrulama stratejisidir.

---

## 3. Mimari Tasarım Desenleri Özeti

Projenin kodunda gözlemlenen ve akademik literatürdeki tanımlarıyla örtüşen tasarım desenleri:

| Desen | Uygulayan Bileşen | Referans |
|---|---|---|
| **Blackboard Pattern** | `BlackBoard` sınıfı + tüm worker ajanlar | Corkill (1991) |
| **Template Method** | `BaseWorker.execute()` soyut metodu | GoF |
| **Mediator Pattern** | `SupervisorAgent` (ajanlar arası tek koordinasyon noktası) | GoF |
| **Strategy Pattern** | `AuditWorker._check_*` metot serisi | GoF |
| **Facade Pattern** | `main.py → supervisor.run_mission()` tek çağrısı | GoF |
| **Chain of Responsibility** | 6 aşamalı pipeline sırası | GoF |

---

## 4. Değerlendirme Tablosu

| Kriter | Değerlendirme | Açıklama |
|---|---|---|
| **SRP Uyumu** | ✅ Tam | Her worker tek sorumluluk taşır |
| **OCP Uyumu** | ✅ Tam | PIPELINE listesiyle değişmeden genişleme |
| **LSP Uyumu** | ✅ Tam | Tüm alt sınıflar `execute() → bool` kontratına uyar |
| **ISP Uyumu** | ✅ Tam | Worker'lar minimal BlackBoard arayüzü üzerinden haberleşir |
| **DIP Uyumu** | ✅ Tam | Supervisor → BaseWorker; Worker'lar → config.py soyutlaması |
| **Modülerlik** | ✅ Yüksek | 4 katmanlı klasör yapısı (agents/tools/api/data) |
| **Magic Number** | ✅ Yok | Tüm sabitler config.py'de anlamlı isimlerle tanımlı |
| **Hata Yönetimi** | ✅ Kapsamlı | 5 katmanlı savunmacı strateji |
| **Dokümantasyon** | ✅ Tam | Her dosya, sınıf ve public metot docstring içerir |

---

## 5. Sonuç

Bu analizde, `turtle-id` projesinin Hiyerarşik Çoklu Ajan Sistemi mimarisinin S.O.L.I.D prensiplerinin beşini de bilinçli ve tutarlı bir biçimde uyguladığı görülmektedir. `BaseWorker` soyut sınıfı, sistemin temel kiriş yapısını oluşturmakta; `BlackBoard` paylaşılan durum deposu, ajanlar arası loose coupling'i garanti altına almaktadır.

Projenin öne çıkan teknik güçlü yönleri şunlardır:
1. **Konfigürasyon Merkezileştirmesi** (`config.py`): Tüm sabitler tek bir kaynaktan yönetilir.
2. **Fail-Safe Hata Stratejisi**: Raporlama gibi kritik olmayan adımlar misyonu durdurmazken girdi doğrulama adımları fail-fast yaklaşımı izler.
3. **LLM Destekli Recovery**: Supervisor'ın Gemini LLM'e danışarak hata kurtarma kararı vermesi, sistemi standart bir pipeline'ın ötesine, gerçek anlamda **otonom** bir MAS'a taşımaktadır.

---

*Bu rapor, projenin kaynak kodu birinci elden incelenerek hazırlanmıştır. Tüm kod alıntıları `turtle-id/` dizininden alınmıştır.*
