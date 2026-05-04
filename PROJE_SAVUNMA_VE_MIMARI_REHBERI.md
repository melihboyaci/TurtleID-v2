# TurtleID-v2 Proje Savunma ve Mimari Rehberi

Bu doküman, `TurtleID-v2` projesinin akademik savunmasında kullanılmak üzere hazırlanmıştır. Amaç; projenin yalnızca çalışan bir görüntü tanıma uygulaması olmadığını, aynı zamanda **Hiyerarşik Çoklu Ajan Mimarisi**, **Shared Blackboard tasarım deseni**, **SOLID prensipleri**, **Clean Code yaklaşımı** ve **Hybrid AI mühendisliği** üzerine kurulu savunulabilir bir yazılım sistemi olduğunu göstermektir.

Doküman hazırlanırken proje içindeki gerçek dosyalar temel alınmıştır:

- `main.py`
- `config.py`
- `blackboard.py`
- `agents/__init__.py`
- `agents/supervisor.py`
- `agents/audit.py`
- `agents/head_detection.py`
- `agents/preprocessing.py`
- `agents/recognition.py`
- `agents/evaluation.py`
- `agents/reporting.py`
- `agents/tensor_utils.py`
- `report_manager.py`
- `kayit_yardimcisi.py`
- `README.md`

---

## 1. Projenin Genel Amacı

`TurtleID-v2`, deniz kaplumbağalarını lateral kafa profili ve göz çevresi pul desenleri üzerinden tanımayı hedefleyen bir kimlik tespit sistemidir. Sistem, sorgu olarak verilen kırpılmış kafa profili görselini Gemini ile doğrular, görseli derin öğrenme modeline uygun hale getirir, ResNet50 tabanlı embedding üretir, veritabanındaki kayıtlı bireylerle karşılaştırır ve sonuç olarak en olası kimliği veya yeni birey durumunu raporlar.

Projenin güçlü tarafı yalnızca görüntü tanıma yapması değildir. Asıl güçlü tarafı, bu tanıma sürecinin mimari olarak kontrollü, modüler, gözlemlenebilir ve genişletilebilir şekilde tasarlanmış olmasıdır.

Sistem şu bileşenleri bir araya getirir:

- **Hiyerarşik çoklu ajan mimarisi**
- **Supervisor Pattern**
- **Shared Blackboard Pattern**
- **OpenCV tabanlı görsel okuma ve normalizasyon**
- **Gemini Vision tabanlı kırpılmış kafa profili doğrulama**
- **ResNet50 tabanlı embedding çıkarımı**
- **Cosine Similarity tabanlı benzerlik ölçümü**
- **Gemini LLM tabanlı hata kurtarma ve raporlama**
- **Dosya tabanlı görev ve gelişim raporları**

---

## 2. Mimari Felsefe: Spagetti Koddan Farkı

Geleneksel bir görüntü işleme uygulamasında görsel okuma, doğrulama, ön işleme, model çağrısı, eşleşme, hata yönetimi ve raporlama çoğu zaman tek bir dosyada veya uzun bir fonksiyonda toplanır. Bu tür yapı zamanla **spagetti kod** haline gelir.

Bunun sonucunda:

- Hata kaynağını bulmak zorlaşır.
- Yeni özellik eklemek riskli hale gelir.
- Test yazmak güçleşir.
- Kodun akademik veya endüstriyel savunulabilirliği azalır.

`TurtleID-v2` bu yaklaşımı terk eder. Her ana sorumluluk ayrı bir worker ajan olarak modellenmiştir:

- `AuditWorker`: Girdi doğrulama
- `HeadDetectionWorker`: Kırpılmış kafa profili için Gemini Vision doğrulaması
- `PreprocessingWorker`: Görseli modele uygun hale getirme
- `RecognitionWorker`: ResNet50 embedding üretimi
- `EvaluationWorker`: Cosine Similarity ile eşleşme kararı
- `ReportingWorker`: Gemini destekli raporlama

Bu ayrım sayesinde sistemde **Separation of Concerns** uygulanmıştır. Her sınıf yalnızca kendi görevinden sorumludur.

Savunmada kullanılabilecek ifade:

> Bu projede görüntü tanıma sürecini tek bir prosedürel akışta toplamak yerine, her uzmanlık alanını bağımsız worker ajanlar olarak modelledim. Böylece sistem spagetti koddan uzak; modüler, test edilebilir ve genişletilebilir bir mimariye kavuştu.

---

## 3. Basit Pipeline'dan Farkı

Sistem dışarıdan şu pipeline gibi görünebilir:

```text
Audit → Head Detection → Preprocessing → Recognition → Evaluation → Reporting
```

Ancak proje yalnızca basit ve pasif bir pipeline değildir. Basit pipeline yaklaşımında adımlar genellikle birbirini doğrudan çağırır. Bu durumda bir adımın implementasyon detayı diğer adımları etkileyebilir.

`TurtleID-v2` ise şu farklara sahiptir:

- Worker ajanlar birbirini doğrudan çağırmaz.
- Tüm veri alışverişi `BlackBoard` üzerinden yapılır.
- `SupervisorAgent` her adımı bilinçli şekilde delege eder.
- Her adımın sonucu loglanır.
- Başarısızlık durumunda Gemini destekli recovery kararı alınır.
- Görev durumu `PENDING`, `RUNNING`, `SUCCESS`, `FAILED` gibi açık durumlarla takip edilir.

Bu nedenle sistem, yalnızca sıralı bir işlem hattı değil; durum farkındalığı olan, hata durumunda karar verebilen, gözlemlenebilir bir **Hiyerarşik Çoklu Ajan Sistemi**dir.

---

## 4. Hiyerarşik Çoklu Ajan Mimarisi

Projede mimari yapı `README.md` içinde şu şekilde özetlenir:

```text
SupervisorAgent
   ├── AuditWorker            (girdi doğrulama)
   ├── HeadDetectionWorker    (Gemini ile kırpılmış kafa profili doğrulama)
   ├── PreprocessingWorker    (224x224 normalize)
   ├── RecognitionWorker      (ResNet50 embedding)
   ├── EvaluationWorker       (Cosine Similarity)
   └── ReportingWorker        (Gemini LLM raporu)
```

Bu yapıda iki temel katman vardır.

### 4.1. Üst Katman: Supervisor

`agents/supervisor.py` içindeki `SupervisorAgent`, sistemin yönetici ajanıdır. Görevleri şunlardır:

- Ortak `BlackBoard` nesnesini oluşturmak
- Sorgu görselini blackboard'a yazmak
- Worker ajanları başlatmak
- Worker ajanları doğru sırada çalıştırmak
- Her adımın sonucunu kontrol etmek
- Hata durumunda recovery kararı almak
- Görev loglarını kaydetmek
- Görev sonunda blackboard'u döndürmek

### 4.2. Alt Katman: Worker Ajanlar

Worker ajanlar uzmanlaşmış alt görevleri yerine getirir. Her worker kendi görevini yapar, çıktısını `BlackBoard` üzerine yazar ve `True` veya `False` döndürür.

Bu mimari, gerçek dünyadaki bir ekip organizasyonuna benzer:

- Supervisor proje yöneticisi gibidir.
- Worker ajanlar uzman ekip üyeleri gibidir.
- Blackboard ortak proje panosu gibidir.

---

## 5. Supervisor Pattern Nasıl Çalışıyor?

`SupervisorAgent.__init__` içinde önce ortam değişkenleri yüklenir, ardından ortak blackboard oluşturulur:

```python
self.bb = BlackBoard()
self.bb.query_image_path = image_path
```

Bu noktada sistemin görev bağlamı oluşur. Sorgu görselinin yolu artık tüm ajanların erişebileceği ortak durum alanındadır.

Sonra tüm worker ajanlar aynı `BlackBoard` referansı ile başlatılır:

```python
self.workers = {
    "audit": AuditWorker(self.bb),
    "head_detection": HeadDetectionWorker(self.bb),
    "preprocessing": PreprocessingWorker(self.bb),
    "recognition": RecognitionWorker(self.bb),
    "evaluation": EvaluationWorker(self.bb),
    "reporting": ReportingWorker(self.bb),
}
```

Buradaki kritik karar şudur: Her worker kendi ayrı state'ine sahip değildir. Tüm worker'lar aynı görev belleğini paylaşır.

### 5.1. Pipeline Sırası

`SupervisorAgent` içinde worker çalıştırma sırası `PIPELINE` sabiti ile belirlenmiştir:

```python
PIPELINE = [
    "audit",
    "head_detection",
    "preprocessing",
    "recognition",
    "evaluation",
    "reporting",
]
```

Bu sıra biyometrik görüntü tanıma açısından mantıksal bir akışı temsil eder:

1. Önce girdi güvenilir mi kontrol edilir.
2. Sonra kafa bölgesi bulunur.
3. Görsel modele uygun hale getirilir.
4. Embedding çıkarılır.
5. Veritabanı ile benzerlik hesaplanır.
6. Sonuç raporlanır.

### 5.2. `delegate()` Metodu

Supervisor'ın en önemli davranışlarından biri `delegate()` metodudur. Bu metod ilgili worker'ı çalıştırır ve sonucunu değerlendirir.

Akış:

1. Worker adı alınır.
2. İlgili worker nesnesi bulunur.
3. Blackboard'da aktif adım güncellenir.
4. Worker çalıştırılır.
5. Başarı veya hata loglanır.
6. Sonuç döndürülür.

Bu, Supervisor Pattern'ın özüdür. Supervisor işi kendisi yapmaz; işi ilgili uzmana delege eder.

Savunmada kullanılabilecek ifade:

> `SupervisorAgent`, alt görevlerin implementasyon detaylarına girmez. Her görevi ilgili worker ajana delege eder. Bu sayede üst seviye akış yönetimi ile alt seviye uzmanlık işlemleri birbirinden ayrılmıştır.

### 5.3. `run_mission()` Metodu

`run_mission()` tüm görevin yaşam döngüsünü yönetir.

Bu metot:

- Görevi `RUNNING` durumuna alır.
- Her pipeline adımını sırayla çalıştırır.
- Başarısız adım olursa Gemini'ye danışır.
- Devam kararı alınırsa akışı sürdürür.
- Durma kararı alınırsa görevi sonlandırır.
- Tüm adımlar başarılıysa görevi `SUCCESS` yapar.
- Görev logunu dosyaya yazar.

Bu metot sayesinde sistem, körü körüne çalışan bir pipeline olmaktan çıkar. Her aşamada görev durumunu bilen, hata olduğunda değerlendirme yapan bir yönetim katmanına sahip olur.

---

## 6. Gemini ile Otonom Hata Kurtarma

`SupervisorAgent`, bir worker başarısız olduğunda `_consult_gemini_for_recovery()` metodunu çağırır. Bu metod Gemini'ye şu bilgileri verir:

- Hangi adımın başarısız olduğu
- Blackboard'daki hata mesajı
- Devam mı edilmeli, durulmalı mı sorusu

Prompt formatı özellikle kısıtlanmıştır:

```text
KARAR: DEVAM veya DUR
NEDEN: (tek cümle Türkçe)
```

Bu iyi bir mühendislik kararıdır. Çünkü LLM çıktısının serbest ve kontrolsüz olmasını engeller. Supervisor, Gemini cevabını parse ederek `continue` kararını çıkarır.

Eğer Gemini'ye erişilemezse sistem güvenli varsayıma döner:

```python
return {"continue": False, "reason": "Gemini'ye ulaşılamadı, güvenli duruş."}
```

Bu yaklaşım **fail-safe** mantığına uygundur. Yani sistem belirsizlik durumunda kontrolsüz devam etmek yerine güvenli şekilde durur.

Önemli nokta: Gemini burada ana tanıma motoru değildir. Gemini, sistemin hata bağlamını yorumlayan karar destek katmanıdır.

Savunmada kullanılabilecek ifade:

> Gemini'yi kimlik tanıma kararının yerine koymadım. Supervisor seviyesinde, hata durumlarında bağlamı yorumlayan bir recovery mekanizması olarak kullandım. Böylece sistem hem deterministik karar mekanizmasını korudu hem de LLM tabanlı otonom hata değerlendirme yeteneği kazandı.

---

## 7. Shared Blackboard Tasarım Deseni

`blackboard.py` dosyasında `BlackBoard` adlı bir `dataclass` bulunur. Bu sınıf, tüm ajanların ortak okuyup yazdığı merkezi durum deposudur.

Temel alanlar:

```python
query_image_path: str = ""
audit_result: dict = field(default_factory=dict)
head_crop: Optional[np.ndarray] = None
head_confidence: float = 0.0
processed_image: Optional[np.ndarray] = None
query_embedding: Optional[np.ndarray] = None
db_embeddings: list = field(default_factory=list)
db_files: list = field(default_factory=list)
match_result: dict = field(default_factory=dict)
current_step: str = "IDLE"
mission_status: str = "PENDING"
error_message: str = ""
mission_log: list = field(default_factory=list)
```

Bu alanlar üç gruba ayrılır:

- **Görev girdisi**: `query_image_path`
- **Ajan çıktıları**: `audit_result`, `head_crop`, `processed_image`, `query_embedding`, `db_embeddings`, `match_result`
- **Misyon durumu**: `current_step`, `mission_status`, `error_message`, `mission_log`

Blackboard deseni, birden fazla uzman modülün ortak bir bilgi alanı üzerinde çalıştığı mimari yaklaşımdır. Her ajan gerekli girdiyi blackboard'dan okur, kendi uzmanlık işlemini yapar ve sonucu tekrar blackboard'a yazar.

---

## 8. Ajanlar Blackboard Üzerinden Nasıl Haberleşiyor?

Bu projede ajanlar birbirlerini doğrudan çağırmaz. Bunun yerine ortak blackboard üzerinden dolaylı iletişim kurarlar.

- `AuditWorker`, `query_image_path` okur ve `audit_result` yazar.
- `HeadDetectionWorker`, `query_image_path` okur; `head_crop` ve `head_confidence` yazar.
- `PreprocessingWorker`, `head_crop` okur ve `processed_image` yazar.
- `RecognitionWorker`, `processed_image` okur; `query_embedding`, `db_embeddings`, `db_files` yazar.
- `EvaluationWorker`, embedding alanlarını okur ve `match_result` yazar.
- `ReportingWorker`, `match_result` ve `mission_log` okur; rapor üretir.

Bu yapı sayesinde worker'lar birbirlerinin sınıf isimlerini, metotlarını veya implementasyon detaylarını bilmez.

**Ajanlar Arası Doğrudan Bağımlılığın Önlenmesi:**
Ajanlar arası bağımlılığı tamamen kaldırmak için, `PreprocessingWorker` ve `RecognitionWorker`'ın ortak ihtiyaç duyduğu matematiksel tensör dönüşüm mantığı `agents/tensor_utils.py` adlı bağımsız bir utility modülüne taşınmıştır. Böylece ajanlar birbirlerini import etmek yerine, bu bağımsız yardımcı modülü kullanırlar. (Dependency Inversion ve DRY prensipleri).

Savunmada kullanılabilecek ifade:

> Blackboard, ajanlar arasında doğrudan bağımlılığı kaldıran ortak bir görev belleğidir. Her ajan yalnızca blackboard üzerinde kendi ilgilendiği alanları okur ve kendi çıktısını yazar. Ayrıca ortak matematiksel operasyonları bağımsız utility modüllerine (örn. tensor_utils) taşıyarak worker'lar arası tight coupling durumunu tamamen ortadan kaldırdım.

---

## 9. Loose Coupling'in Kazanımları

Ajanların doğrudan birbirine bağlı olmaması sisteme ciddi avantaj sağlar.

### 9.1. Değiştirilebilirlik

`HeadDetectionWorker` bugün Gemini Vision ile kırpılmış kafa profilini doğruluyor. Gelecekte farklı bir doğrulayıcı model, lokal sınıflandırıcı veya başka bir LLM ile değiştirilebilir. `PreprocessingWorker` bundan etkilenmez. Çünkü yalnızca `bb.head_crop` alanına bakar.

### 9.2. Test Edilebilirlik

Bir worker test edilirken önce blackboard'a gerekli sahte veri yazılabilir. Örneğin `EvaluationWorker` test edilecekse gerçek ResNet50 çalıştırmaya gerek yoktur. Sahte `query_embedding` ve `db_embeddings` verileri blackboard'a yazılarak değerlendirme mantığı test edilebilir.

### 9.3. Gözlemlenebilirlik

Tüm ara çıktılar blackboard üzerinde bulunduğu için sistemin hangi aşamada ne ürettiği izlenebilir. Bu, akademik savunma açısından önemli bir açıklanabilirlik sağlar.

### 9.4. Genişletilebilirlik

Yeni bir ajan sisteme eklendiğinde mevcut ajanların çoğunun değişmesi gerekmez. Yeni ajan blackboard'dan gerekli veriyi okur ve kendi sonucunu yazar.

### 9.5. Hata Yönetimi

Her ajan hata durumunda `bb.fail()` çağırarak ortak görev durumunu günceller. Böylece hata bilgisi merkezi hale gelir.

---

## 10. Görev Loglama ve İzlenebilirlik

`BlackBoard` içinde `log()` metodu bulunur:

```python
def log(self, agent_name: str, message: str) -> None:
    from datetime import datetime
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_name}] {message}"
    self.mission_log.append(entry)
    print(entry)
```

Bu metot sayesinde:

- Her log zaman damgası alır.
- Hangi ajanın log yazdığı bellidir.
- Loglar `mission_log` listesinde birikir.
- Aynı anda konsola yazdırılır.

`SupervisorAgent._save_mission_log()` bu logları `logs/mission_log.md` dosyasına yazar. `report_manager.py` ise blackboard verisinden `logs/mission_report.md` üretir. `ReportingWorker` ayrıca Gemini ile `gelisim_raporu.md` dosyasına yorumlayıcı gelişim raporu ekler.

Bu sistemde üç raporlama/izleme katmanı vardır:

- **Operasyonel görev logu**: `logs/mission_log.md`
- **Deterministik görev özeti**: `logs/mission_report.md`
- **LLM destekli gelişim raporu**: `gelisim_raporu.md`

---

## 11. SOLID Prensipleri

### 11.1. `BaseWorker` Soyut Sınıfı

`agents/__init__.py` içinde `BaseWorker` soyut sınıfı vardır:

```python
class BaseWorker(ABC):
    def __init__(self, blackboard: BlackBoard):
        self.bb = blackboard

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def execute(self) -> bool:
        pass

    def log(self, message: str) -> None:
        self.bb.log(self.name, message)
```

Bu sınıf, tüm worker ajanların ortak kontratını belirler.

Bir sınıf worker olmak istiyorsa:

- `BaseWorker` sınıfından türemelidir.
- `execute()` metodunu implemente etmelidir.
- Blackboard referansı üzerinden çalışmalıdır.

### 11.2. Dependency Inversion Principle

Dependency Inversion Principle, üst seviye modüllerin alt seviye detaylara değil, soyutlamalara bağımlı olması gerektiğini söyler.

Bu projede:

- Üst seviye modül: `SupervisorAgent`
- Alt seviye modüller: Worker ajanlar
- Soyutlama: `BaseWorker`

Her worker `execute()` kontratına uyar. Supervisor için önemli olan worker'ın iç algoritması değil, bu kontratı sağlamasıdır.

Supervisor şu detayları bilmek zorunda değildir:

- `AuditWorker` dosya boyutunu nasıl kontrol ediyor?
- `HeadDetectionWorker` Gemini mi yoksa lokal bir doğrulayıcı model mi kullanıyor?
- `RecognitionWorker` ResNet50 mi, EfficientNet mi kullanıyor?
- `ReportingWorker` hangi prompt ile rapor yazıyor?

Supervisor yalnızca şunu bilir:

```python
success = worker.execute()
```

Savunmada kullanılabilecek ifade:

> `BaseWorker`, tüm worker ajanlar için ortak bir soyutlama katmanı sağlar. Supervisor, worker'ların iç implementasyonuna değil, `execute()` kontratına bağımlıdır. Bu da Dependency Inversion prensibini destekler.

### 11.3. Open/Closed Principle

Open/Closed Principle, yazılım bileşenlerinin genişlemeye açık, değişikliğe kapalı olması gerektiğini söyler.

Bu projede yeni bir worker eklemek için:

1. `BaseWorker` sınıfından türeyen yeni bir sınıf oluşturulur.
2. `execute()` metodu yazılır.
3. Worker, `SupervisorAgent.workers` sözlüğüne eklenir.
4. Gerekirse `PIPELINE` listesine yeni adım eklenir.

Gelecekte eklenebilecek ajan örnekleri:

- `SpeciesClassificationWorker`
- `ImageQualityWorker`
- `DuplicateDetectionWorker`
- `HumanReviewWorker`
- `DatasetAugmentationWorker`

Mevcut worker'ların iç kodu bozulmadan sistem genişletilebilir. Bu nedenle proje Open/Closed prensibine uygundur.

### 11.4. Single Responsibility Principle

Her worker tek bir sorumluluğa sahiptir:

- `AuditWorker`: Dosya doğrulama
- `HeadDetectionWorker`: Kafa bölgesi tespiti ve doğrulama
- `PreprocessingWorker`: Görüntü normalizasyonu
- `RecognitionWorker`: Embedding çıkarımı
- `EvaluationWorker`: Benzerlik hesabı ve sınıflandırma
- `ReportingWorker`: Raporlama

Bu görev ayrımı, Single Responsibility Principle'ın açık bir uygulamasıdır.

### 11.5. Kod İçi SOLID Belgelendirmesi

Gerçekleştirilen mimari refactoring sonucunda, tüm `agents` modüllerinde, `config.py` ve `blackboard.py` dosyalarında modül başına açıklayıcı bir "SOLID / Clean Code Uyum Notu" eklenmiştir. Bir hoca veya jüri üyesi doğrudan kodu okuduğunda, o dosyanın mimarideki rolünü ve hangi SOLID kurallarına uyduğunu anında görebilir.

---

## 12. Clean Code Yaklaşımı

Projede Clean Code açısından dikkat çeken noktalar şunlardır:

### 12.1. Anlamlı Sınıf İsimleri

Sınıf isimleri doğrudan sorumluluğu ifade eder:

- `SupervisorAgent`
- `AuditWorker`
- `HeadDetectionWorker`
- `PreprocessingWorker`
- `RecognitionWorker`
- `EvaluationWorker`
- `ReportingWorker`

### 12.2. Anlamlı Veri Alanları

Blackboard alanları açık isimlendirilmiştir:

- `query_image_path`
- `head_crop`
- `processed_image`
- `query_embedding`
- `db_embeddings`
- `match_result`
- `mission_status`
- `error_message`

Bu alanlar sistemin veri akışını okunabilir hale getirir.

### 12.3. Küçük ve Odaklı Metotlar

Örneğin `AuditWorker` içinde doğrulama adımları ayrı metotlara ayrılmıştır:

- `_check_extension()`
- `_check_size()`
- `_check_readable()`
- `_check_dimensions()`

Bu yapı hem okunabilirliği hem test edilebilirliği artırır.

### 12.4. Sabitlerin Merkezi Yönetimi (`config.py`)

Daha önce kod içine dağılmış olan sihirli sayılar (magic numbers), dosya yolları ve yapılandırma ayarları, `config.py` modülü altında merkezileştirilmiştir. 

```python
GEMINI_MODEL_NAME = "gemini-2.5-flash"
MATCH_THRESHOLD = 0.85
POSSIBLE_THRESHOLD = 0.70
TARGET_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 10
```

Bu sayede, Single Responsibility Principle ve Open/Closed Principle desteklenmiştir; bir ayar değiştiğinde kod değil, sadece konfigürasyon dosyası güncellenir. `load_dotenv()` gibi işlemler de sadece başlangıç noktalarında (Supervisor ve Config) yapılarak DRY ihlali engellenmiştir.

### 12.5. Merkezi Hata Yönetimi

Her worker hata durumunda `bb.fail()` çağırır. Bu merkezi hata yönetimi, sistemin tutarlı davranmasını sağlar.

---

## 13. Hybrid AI Mimarisi

Bu projede her şey LLM'e bırakılmamıştır. Bunun yerine deterministik bilgisayarlı görü ve derin öğrenme yöntemleri, LLM tabanlı semantik ve bağlamsal yeteneklerle harmanlanmıştır.

Bu yaklaşım **Hybrid AI** olarak değerlendirilebilir.

---

## 14. Neden Her Şeyi LLM'e Bırakmadık?

Bir kaplumbağa görselini doğrudan LLM'e gönderip “Bu kim?” diye sormak teknik olarak mümkün gibi görünebilir. Ancak bu yaklaşım akademik ve mühendislik açısından zayıftır.

Biyometrik kimlik tespiti için şu özellikler önemlidir:

- Tekrarlanabilir sonuç
- Ölçülebilir benzerlik skoru
- Açıklanabilir karar süreci
- Veritabanı ile sistematik karşılaştırma
- Eşik tabanlı sınıflandırma
- Model çıktısının kontrol edilebilir olması

LLM'ler aynı girdiye farklı yanıtlar verebilir. Ayrıca LLM'in verdiği kimlik kararı doğrudan sayısal ve denetlenebilir bir benzerlik ölçüsüne dayanmayabilir.

Bu nedenle projede LLM ana tanıma motoru olarak değil, yardımcı zeka katmanı olarak kullanılmıştır.

Savunmada kullanılabilecek ifade:

> Kimlik tespit kararını tamamen LLM'e bırakmadım. Çünkü biyometrik tanımada ölçülebilir, tekrar edilebilir ve açıklanabilir skorlar gerekir. Bu nedenle ana karşılaştırma mekanizmasını ResNet50 embedding ve cosine similarity üzerine kurdum. Gemini'yi ise semantik doğrulama, hata kurtarma ve raporlama gibi bağlamsal görevlerde kullandım.

---

## 15. Deterministik ve Sayısal Bileşenler

### 15.1. OpenCV

OpenCV, sistemde görsel okuma, renk uzayı dönüşümü ve yeniden boyutlandırma gibi deterministik işlemler için kullanılır. Kafa bölgesi için otomatik bounding box çıkarımı yapılmaz; kullanıcı sisteme zaten kırpılmış kafa profili görseli verir.

İşlem adımları:

1. Görsel okunur.
2. RGB formatına çevrilir.
3. Gemini ile kafa profili doğrulanır.
4. Doğrulanan görsel `224x224` boyutuna getirilir.

Bu ayrım sayesinde geometrik tespit hataları azaltılır; sistemin görsel tanıma kısmı kullanıcı kontrollü kırpım ve Gemini doğrulaması üzerine kurulur.

### 15.2. ResNet50

`RecognitionWorker`, TensorFlow/Keras üzerinden ResNet50 modelini kullanır:

```python
ResNet50(weights='imagenet', include_top=False, pooling='avg')
```

Burada ResNet50 sınıflandırıcı olarak değil, feature extractor olarak kullanılır.

- `include_top=False`: Son sınıflandırma katmanı çıkarılır.
- `pooling='avg'`: Görselden sabit boyutlu embedding vektörü elde edilir.

Bu embedding vektörü, kaplumbağanın görsel imzası gibi değerlendirilir.

### 15.3. Cosine Similarity

`EvaluationWorker`, sorgu embedding'i ile veritabanındaki embedding'leri cosine similarity ile karşılaştırır:

```python
score = 1 - cosine(self.bb.query_embedding, db_emb)
```

Karar eşikleri:

| Skor      | Durum           |
| --------- | --------------- |
| `>= 0.85` | `GÜÇLÜ_EŞLEŞME` |
| `>= 0.70` | `OLASI_EŞLEŞME` |
| `< 0.70`  | `YENİ_BİREY`    |

Bu eşikler sayesinde sistemin kararı açıklanabilir hale gelir.

---

## 16. LLM Kullanılan Bileşenler

### 16.1. Gemini Vision ile Kafa Doğrulama

`HeadDetectionWorker`, OpenCV ile elde edilen crop sonucunu Gemini Vision'a doğrulatır.

Prompt şunu sorar:

```text
Bu görselde bir deniz kaplumbağasının lateral (yan) kafa profili görünüyor mu?
SONUÇ: EVET veya HAYIR
AÇIKLAMA: (tek cümle Türkçe)
```

Görev paylaşımı şöyledir:

- OpenCV aday bölgeyi üretir.
- Gemini Vision aday bölgenin semantik olarak kaplumbağa kafa profili olup olmadığını yorumlar.

Bu güçlü bir hybrid tasarımdır. Çünkü klasik görüntü işleme şekil ve kenar bilgisiyle çalışırken, Gemini Vision semantik anlamı değerlendirebilir.

### 16.2. Gemini ile Supervisor Recovery

Bir worker başarısız olduğunda Supervisor Gemini'ye danışır. Burada amaç, hatanın kurtarılabilir olup olmadığını yorumlamaktır.

Gemini yanıtı şu şekilde beklenir:

```text
KARAR: DEVAM
NEDEN: ...
```

veya:

```text
KARAR: DUR
NEDEN: ...
```

Bu, LLM'in kontrolsüz metin üretmesini değil, sınırlı ve parse edilebilir karar üretmesini sağlar.

### 16.3. Gemini ile Raporlama

`ReportingWorker`, görev sonucunu Gemini'ye vererek doğal dilde gelişim raporu üretir.

Prompt içinde şu bilgiler bulunur:

- Tespit edilen kimlik
- Benzerlik skoru
- Eşleşme durumu
- Sistem logları

Gemini'den şu formatta rapor istenir:

- Ne yapıldı?
- Sonuç ne?
- Problemler neler?
- İyileştirme önerileri neler?

Bu sayede sistem teknik çıktısını insan tarafından okunabilir hale getirir.

---

## 17. `kayit_yardimcisi.py` ve Veritabanı Oluşturma

`kayit_yardimcisi.py`, ana tanıma pipeline'ının dışında yardımcı bir kayıt aracıdır. Veritabanına yeni kaplumbağa eklemek için kullanılır.

Bu dosyada Gemini Vision şu amaçlarla kullanılır:

- Tür tespiti
- Profil yönü tahmini
- Güven seviyesi üretimi
- Metadata notu oluşturma

Kayıt yapısı şu şekildedir:

```text
data/database/
└── kaplumbaga_adi/
    ├── sag_profil.jpg
    ├── sol_profil.jpg
    └── metadata.json
```

`RecognitionWorker`, bu klasör yapısını kullanarak her birey için sağ ve sol profil embedding'lerini çıkarır.

Bireylerin her görselinden ayrı ayrı embedding vektörleri çıkarılır ve bir liste olarak (çoğul `embeddings`) saklanır.

`EvaluationWorker` bu embedding'leri **Max-of-Images (Galeri/Probe)** yaklaşımıyla değerlendirir:
Sorgu görseli (probe), o bireye ait (galerideki) tüm görsellerle ayrı ayrı karşılaştırılır ve **en yüksek benzerlik skoru (max)** o bireyin nihai skoru olarak kabul edilir.

Bu yaklaşım, eski "ortalama alma (averaging)" yönteminden kaynaklanabilecek vektörel bulanıklığı (blur) önler.

Savunmada kullanılabilecek ifade:

> Veritabanı yapısında her birey için sağ ve sol profiller ayrı tutulur. Sistemin eski halinde bu profillerin ortalaması alınıyordu, ancak yeni mimaride Max-of-Images (Galeri/Probe) yaklaşımına geçtik. RecognitionWorker her görselin embedding'ini ayrı ayrı saklar, EvaluationWorker ise sorgu görseli ile bu görseller arasındaki en yüksek benzerliği o bireyin skoru olarak kabul eder. Bu sayede averaging-blur problemini önlemiş olduk.

---

## 18. Uçtan Uca Sistem Akışı

### 18.1. `main.py`

Sistem `main.py` üzerinden başlar.

Akış:

1. `data/query` klasöründen ilk görsel alınır.
2. `SupervisorAgent` oluşturulur.
3. `run_mission()` çalıştırılır.
4. Dönen blackboard rapor yöneticisine verilir.
5. Sonuç ekrana yazdırılır.

`main.py`, iş mantığını bilmez. Sadece uygulama giriş noktasıdır. Bu da mimari ayrımı güçlendirir.

### 18.2. Audit

`AuditWorker`, görselin geçerli olup olmadığını kontrol eder.

Kontroller:

- Uzantı `.jpg`, `.jpeg`, `.png` mi?
- Dosya var mı?
- Dosya boyutu 10 MB altında mı?
- PIL ile okunabiliyor mu?
- Minimum boyut 100x100 piksel mi?

### 18.3. Head Detection

`HeadDetectionWorker` görselden kafa bölgesi çıkarmaya çalışmaz. Kullanıcının sisteme kırpılmış kafa profili yüklediği varsayılır. Worker, görseli Gemini Vision'a göndererek net bir deniz kaplumbağası yan kafa profili olup olmadığını Evet/Hayır mantığıyla doğrular.

Başarılı olursa şu alanlar dolar:

- `bb.head_crop`
- `bb.head_confidence`

### 18.4. Preprocessing

`PreprocessingWorker`, `head_crop` alanını okur ve görüntüyü `224x224` boyutuna getirir.

Başarılı olursa:

- `bb.processed_image`

alanı dolar.

### 18.5. Recognition

`RecognitionWorker`, sorgu görselinden ve veritabanı görsellerinden embedding çıkarır.

Başarılı olursa:

- `bb.query_embedding`
- `bb.db_embeddings`
- `bb.db_files`

alanları dolar.

### 18.6. Evaluation

`EvaluationWorker`, cosine similarity ile en yakın eşleşmeyi bulur.

Sonucu şu yapıda blackboard'a yazar:

```python
self.bb.match_result = {
    "name": best_name,
    "score": best_score,
    "status": status,
    "profile_note": "Max-of-images yaklaşımı: galeri bazlı en yüksek benzerlik kullanıldı",
}
```

### 18.7. Reporting

`ReportingWorker`, Gemini ile gelişim raporu üretir ve `gelisim_raporu.md` dosyasına ekler.

`ReportManager` ise görev özetini `logs/mission_report.md` dosyasına yazar.

---

## 19. Akademik Savunmada Öne Çıkarılacak Güçlü Yönler

### 19.1. Mimari Olgunluk

Proje tek dosyalık bir demo değil; görevlerin ayrıldığı, ajanların koordine edildiği, ortak durum yönetiminin bulunduğu mimari bir sistemdir.

### 19.2. Açıklanabilirlik

Sistem yalnızca sonuç vermez. Ara adımlar, loglar, skorlar ve raporlar üretir. Bu, akademik değerlendirme açısından önemlidir.

### 19.3. Hybrid AI Tasarımı

LLM, ResNet50 ve OpenCV doğru görevlerde kullanılmıştır. Her teknoloji güçlü olduğu alanda konumlandırılmıştır.

### 19.4. Genişletilebilirlik

Yeni worker ajanlar eklenebilir. Mevcut sistem bozulmadan yeni yetenekler kazandırılabilir.

### 19.5. Hata Yönetimi

Hatalar merkezi olarak blackboard'a yazılır. Supervisor hata durumunda Gemini'ye danışabilir. Gemini erişilemezse sistem güvenli şekilde durur.

### 19.6. Temiz Kod ve SOLID

Her sınıfın sorumluluğu ayrıdır. `BaseWorker` ortak kontrat sağlar. Sistem, Open/Closed ve Dependency Inversion prensiplerini destekler.

---

## 20. Jüri İçin Kısa Savunma Metni

Aşağıdaki metin savunmada doğrudan kullanılabilir:

> TurtleID-v2 projesini klasik bir sıralı görüntü işleme betiği olarak değil, hiyerarşik çoklu ajan mimarisiyle tasarladım. Sistemin merkezinde `SupervisorAgent` yer alır. Supervisor, her uzman worker ajanı sırayla görevlendirir ve tüm görev durumunu `BlackBoard` üzerinden takip eder.

> Worker ajanlar birbirlerini doğrudan çağırmaz. Bunun yerine ortak blackboard üzerinden veri paylaşırlar. Bu sayede sistem gevşek bağlı, modüler ve genişletilebilir hale gelir. Örneğin kafa tespit algoritmasını değiştirmek istersem, yalnızca `HeadDetectionWorker` sınıfını değiştirmem yeterlidir; diğer ajanlar `bb.head_crop` alanını okumaya devam eder.

> Kimlik tespit tarafında tamamen LLM tabanlı bir yaklaşım kullanmadım. Çünkü biyometrik tanımada ölçülebilir, tekrar edilebilir ve açıklanabilir skorlar gerekir. Bu nedenle ana karşılaştırma mekanizmasını ResNet50 embedding ve cosine similarity üzerine kurdum. Gemini'yi ise semantik doğrulama, hata kurtarma ve raporlama gibi bağlamsal görevlerde kullandım.

> SOLID açısından bakıldığında, tüm worker ajanlar `BaseWorker` soyut sınıfından türemektedir. Bu sınıf `execute()` kontratını zorunlu kılar. Böylece Supervisor, worker'ların iç detaylarına değil ortak arayüze bağımlı olur. Yeni bir ajan eklemek için mevcut ajanları bozmak gerekmez; `BaseWorker`'dan türeyen yeni bir sınıf eklenip pipeline'a dahil edilebilir.

> Sonuç olarak TurtleID-v2 yalnızca çalışan bir görüntü tanıma uygulaması değil; yapay zeka, yazılım mimarisi, çoklu ajan sistemleri ve clean code prensiplerini bir araya getiren savunulabilir bir mühendislik projesidir.

---

## 21. Sonuç

`TurtleID-v2` projesi şu mimari niteliklere sahiptir:

- Gerçek hiyerarşik çoklu ajan organizasyonu
- Supervisor merkezli görev koordinasyonu
- Shared Blackboard ile gevşek bağlı ajan iletişimi
- SOLID uyumlu worker kontratı
- Deterministik ResNet50 + cosine similarity karar mekanizması
- Gemini Vision ile semantik doğrulama
- Gemini LLM ile hata kurtarma ve raporlama
- Merkezi loglama ve rapor üretimi
- Genişletilebilir klasör/veritabanı yapısı

Bu yönleriyle proje, akademik savunmada hem yapay zeka uygulaması hem de yazılım mimarisi projesi olarak güçlü şekilde açıklanabilir.
