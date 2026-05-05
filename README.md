# 🐢 TurtleID: Çoklu Ajan Mimarisi ile Otonom Deniz Kaplumbağası Tanıma Sistemi

`TurtleID`, deniz kaplumbağalarını lateral (yan) kafa profili ve göz çevresi pul desenleri üzerinden yüksek doğrulukla tanımayı hedefleyen, **Hiyerarşik Çoklu Ajan Sistemi (MAS)** ve **Shared Blackboard (Ortak Kara Tahta)** tasarım desenleri üzerine inşa edilmiş modern bir yapay zeka ve görüntü işleme sistemidir.

Proje yalnızca bir derin öğrenme modeli çalıştırmakla kalmaz; girdi doğrulama, ön işleme, vektörel çıkarım, karar mekanizması ve raporlama süreçlerinin her birini "Single Responsibility (Tek Sorumluluk)" prensibiyle çalışan otonom uzman ajanlara (worker) delege ederek spagetti koddan uzak, akademik standartlarda savunulabilir bir yazılım mimarisi sunar.

---

## 🌟 Temel Özellikler (Key Features)

- **Hiyerarşik Çoklu Ajan Sistemi (MAS):** Görüntü işleme ve tanıma süreci, tek bir monolitik akış yerine birbirinden izole edilmiş, birbirini tanımayan **6 uzman ajan** ve bunları koordine eden **1 yönetici (Supervisor)** ile yönetilir. Ajanlar arası haberleşme `BlackBoard` üzerinden sağlanır (Loose Coupling).
- **Özel Siyam Ağı (Triplet Loss):** Hazır ImageNet ağırlıklarıyla yetinilmemiş; proje için özel olarak **Triplet Loss** fonksiyonuyla eğitilmiş, L2 Normalizasyonlu ResNet50 tabanlı spesifik bir embedding uzayı oluşturulmuştur.
- **Max-of-Images Eşleştirme Stratejisi:** Bireyin görsellerinin vektörel ortalamasını (mean) almak yerine, sektör standardı olan **Galeri/Probe** kosinüs benzerliği algoritması (Max-of-Images) kullanılmıştır. Bu sayede "averaging blur" problemi çözülmüş ve **%90+ (T007: %92.8, T003: %97.5)** gibi yüksek eşleşme doğruluk oranlarına ulaşılmıştır.
- **LLM Destekli Doğrulama (Hybrid AI):** Hatalı veya alakasız fotoğrafların (örn. sadece deniz, kabuk) derin öğrenme hattına girmesini önlemek için **Gemini Vision** tabanlı otonom bir Baş Tespiti (Head Detection) onay sistemi entegre edilmiştir.

---

## 🏛️ Sistem Mimarisi ve Ajanlar

Sistem, **Supervisor Pattern** ile yönetilir. `SupervisorAgent`, işi kendi yapmaz; ortak bir `BlackBoard` referansını paylaşan uzman ajanlara (`Worker`) sırayla delege eder ve hata durumunda Gemini LLM'e danışarak otonom kurtarma (recovery) kararı alır.

| Ajan (Worker)              | Mimari Rolü ve Sorumluluğu                                                                                                                                                                                                   |
| :------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 👨‍💼 **SupervisorAgent**     | Çoklu ajan hiyerarşisinin yöneticisi. Ajanları çalıştırır, BlackBoard akışını yönetir ve hata durumunda LLM destekli _recovery_ kararı alır.                                                                                 |
| 📋 **BlackBoard**          | Ajanlar arası veri aktarımını sağlayan merkezi durum (state) nesnesi. Ajanların birbirine doğrudan bağlanmasını (tight coupling) engeller.                                                                                   |
| 🛡️ **AuditWorker**         | Sisteme giren görselin format (.jpg, .png), boyut (MB) ve çözünürlük açısından asgari standartları sağlayıp sağlamadığını doğrular.                                                                                          |
| 🐢 **HeadDetectionWorker** | Girdi görselini Gemini Vision API'sine göndererek, görselde net bir "deniz kaplumbağası yan kafa profili" olup olmadığını doğrular.                                                                                          |
| ⚙️ **PreprocessingWorker** | Doğrulanmış kafa görselini önce boyutlandırır, ardından su altı ışık ve doku bozulmalarını dengelemek için **LAB uzayında CLAHE** (görüntü iyileştirme) uygular ve son olarak modelin beklediği Tensör formatına dönüştürür. |
| 🧠 **RecognitionWorker**   | Özel eğitilmiş ResNet50 modelini kullanarak sorgu görselinden ve veritabanındaki bireylerden 256 boyutlu L2-normalize embedding vektörleri üretir.                                                                           |
| ⚖️ **EvaluationWorker**    | Sorgu vektörü ile veritabanındaki vektörleri `Cosine Similarity` kullanarak _Max-of-Images_ mantığıyla karşılaştırır ve eşleşme (Strong/Possible/New) kararı verir.                                                          |
| 📝 **ReportingWorker**     | `EvaluationWorker`'ın ürettiği deterministik sonucu, Gemini LLM kullanarak analiz eder ve doğal dilde, insan okumasına uygun görev gelişim raporları (`gelisim_raporu.md`) yazar.                                            |

---

## 🚀 Kurulum ve Çalıştırma (Installation & Usage)

### Gereksinimler

Sistem **Python 3.10+** ortamında çalışacak şekilde tasarlanmıştır.

1. Proje bağımlılıklarını yükleyin:
   ```bash
   pip install -r gereksinimler.txt
   ```
2. Çevresel değişkenleri yapılandırın:
   ```bash
   cp .env.example .env
   ```
   _Oluşturulan `.env` dosyasının içine geçerli bir `GEMINI_API_KEY` eklemeyi unutmayın._

> **⚠️ WSL2 / Linux Kullanıcıları İçin Uyarı:**
> TensorRT veya CUDA GPU ivmelendirmesi kullanıyorsanız TensorFlow'un bellek tahsis logları konsolda belirebilir. Bu normal bir durumdur. İşlemci (CPU) modunda çalışmak için komut satırında `export CUDA_VISIBLE_DEVICES="-1"` yapabilirsiniz.

### Çalıştırma

Sistemin artık modern bir web arayüzü (GUI) bulunmaktadır. İşlemlerinizi web tarayıcısı üzerinden interaktif bir şekilde gerçekleştirebilirsiniz.

1. FastAPI sunucusunu başlatın:
   ```bash
   python run_server.py
   ```
2. Tarayıcınızda şu adrese gidin:
   ```text
   http://localhost:8000
   ```
3. Web arayüzü üzerinden deniz kaplumbağası görselini yükleyip, sistemin sağladığı kırpma (crop) aracı ile kafa profilini seçerek **"Kimlik Tespiti"** yapabilirsiniz.

_(Not: Eski CLI yöntemi ile toplu test yapmak isterseniz `data/query/` klasörüne fotoğraf koyarak `python main.py` komutunu kullanmaya devam edebilirsiniz.)_

---

## 🔮 Gelecek Çalışmalar (Future Works)

- **%99 Doğruluk Hedefi:** Domain Shift probleminin çözülmesinin ardından, aydınlatma kalibrasyonu yapılmış veri seti ile Triplet Loss ağının yeniden eğitilmesi ve literatürdeki SOTA (State of the Art) seviyesi olan %99'luk tanımlama eşiğinin yakalanması hedeflenmektedir.
- **Gelişmiş Veritabanı:** JSON tabanlı hiyerarşik dosya sistemi yerine, vektörel aramalarda çok daha hızlı tepki veren Milvus veya Pinecone gibi bir VectorDB entegrasyonu.
