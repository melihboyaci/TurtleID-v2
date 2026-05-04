# 🐢 TurtleID: Çoklu Ajan Mimarisi ile Otonom Deniz Kaplumbağası Tanıma Sistemi

`TurtleID`, deniz kaplumbağalarını lateral (yan) kafa profili ve göz çevresi pul desenleri üzerinden yüksek doğrulukla tanımayı hedefleyen, **Hiyerarşik Çoklu Ajan Sistemi (MAS)** ve **Shared Blackboard (Ortak Kara Tahta)** tasarım desenleri üzerine inşa edilmiş modern bir yapay zeka ve görüntü işleme sistemidir.

Proje yalnızca bir derin öğrenme modeli çalıştırmakla kalmaz; girdi doğrulama, ön işleme, vektörel çıkarım, karar mekanizması ve raporlama süreçlerinin her birini "Single Responsibility (Tek Sorumluluk)" prensibiyle çalışan otonom uzman ajanlara (worker) delege ederek spagetti koddan uzak, akademik standartlarda savunulabilir bir yazılım mimarisi sunar.

---

## 🌟 Temel Özellikler (Key Features)

* **Hiyerarşik Çoklu Ajan Sistemi (MAS):** Görüntü işleme ve tanıma süreci, tek bir monolitik akış yerine birbirinden izole edilmiş, birbirini tanımayan **6 uzman ajan** ve bunları koordine eden **1 yönetici (Supervisor)** ile yönetilir. Ajanlar arası haberleşme `BlackBoard` üzerinden sağlanır (Loose Coupling).
* **Özel Siyam Ağı (Triplet Loss):** Hazır ImageNet ağırlıklarıyla yetinilmemiş; proje için özel olarak **Triplet Loss** fonksiyonuyla eğitilmiş, L2 Normalizasyonlu ResNet50 tabanlı spesifik bir embedding uzayı oluşturulmuştur.
* **Max-of-Images Eşleştirme Stratejisi:** Bireyin görsellerinin vektörel ortalamasını (mean) almak yerine, sektör standardı olan **Galeri/Probe** kosinüs benzerliği algoritması (Max-of-Images) kullanılmıştır. Bu sayede "averaging blur" problemi çözülmüş ve **%90+ (T007: %92.8, T003: %97.5)** gibi yüksek eşleşme doğruluk oranlarına ulaşılmıştır.
* **LLM Destekli Doğrulama (Hybrid AI):** Hatalı veya alakasız fotoğrafların (örn. sadece deniz, kabuk) derin öğrenme hattına girmesini önlemek için **Gemini Vision** tabanlı otonom bir Baş Tespiti (Head Detection) onay sistemi entegre edilmiştir.

---

## 🏛️ Sistem Mimarisi ve Ajanlar

Sistem, **Supervisor Pattern** ile yönetilir. `SupervisorAgent`, işi kendi yapmaz; ortak bir `BlackBoard` referansını paylaşan uzman ajanlara (`Worker`) sırayla delege eder ve hata durumunda Gemini LLM'e danışarak otonom kurtarma (recovery) kararı alır.

| Ajan (Worker) | Mimari Rolü ve Sorumluluğu |
| :--- | :--- |
| 👨‍💼 **SupervisorAgent** | Çoklu ajan hiyerarşisinin yöneticisi. Ajanları çalıştırır, BlackBoard akışını yönetir ve hata durumunda LLM destekli *recovery* kararı alır. |
| 📋 **BlackBoard** | Ajanlar arası veri aktarımını sağlayan merkezi durum (state) nesnesi. Ajanların birbirine doğrudan bağlanmasını (tight coupling) engeller. |
| 🛡️ **AuditWorker** | Sisteme giren görselin format (.jpg, .png), boyut (MB) ve çözünürlük açısından asgari standartları sağlayıp sağlamadığını doğrular. |
| 🐢 **HeadDetectionWorker** | Girdi görselini Gemini Vision API'sine göndererek, görselde net bir "deniz kaplumbağası yan kafa profili" olup olmadığını doğrular. |
| ⚙️ **PreprocessingWorker** | *(Görüntü iyileştirme YAPMAZ)* Doğrulanmış kafa görselini, derin öğrenme modelinin beklediği Tensör formatına (224x224, preprocess_input, batch boyutu) dönüştürür. |
| 🧠 **RecognitionWorker** | Özel eğitilmiş ResNet50 modelini kullanarak sorgu görselinden ve veritabanındaki bireylerden 256 boyutlu L2-normalize embedding vektörleri üretir. |
| ⚖️ **EvaluationWorker** | Sorgu vektörü ile veritabanındaki vektörleri `Cosine Similarity` kullanarak *Max-of-Images* mantığıyla karşılaştırır ve eşleşme (Strong/Possible/New) kararı verir. |
| 📝 **ReportingWorker** | `EvaluationWorker`'ın ürettiği deterministik sonucu, Gemini LLM kullanarak analiz eder ve doğal dilde, insan okumasına uygun görev gelişim raporları (`gelisim_raporu.md`) yazar. |

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
   *Oluşturulan `.env` dosyasının içine geçerli bir `GEMINI_API_KEY` eklemeyi unutmayın.*

> **⚠️ WSL2 / Linux Kullanıcıları İçin Uyarı:**
> TensorRT veya CUDA GPU ivmelendirmesi kullanıyorsanız TensorFlow'un bellek tahsis logları konsolda belirebilir. Bu normal bir durumdur. İşlemci (CPU) modunda çalışmak için komut satırında `export CUDA_VISIBLE_DEVICES="-1"` yapabilirsiniz.

### Çalıştırma

Sistem sorgu (test) edilecek görselleri belirli bir klasörde bekler.

1. Test etmek istediğiniz **kırpılmış kafa profili (lateral) görselini** `data/query/` klasörünün içine yerleştirin (örn. `data/query/test1.jpg`).
2. Ana pipeline'ı başlatın:
   ```bash
   python main.py
   ```
3. Sistem aşama aşama (loglayarak) çalışacak ve sonucunda hem ekrana nihai kararı (Kimlik, Eşleşme Oranı, Durum) yazdıracak hem de `logs/mission_log.md` ve `gelisim_raporu.md` dosyalarını güncelleyecektir.

---

## 🔮 Gelecek Çalışmalar (Future Works)

* **Domain Shift (Veri Dağılımı Uyuşmazlığı) Çözümü:** Şu anda mevcut model RGB renk uzayındaki ham görsellerle eğitilmiştir. İlerleyen çalışmalarda, su altı çekimlerindeki renk ve ışık bozulmalarını gidermek amacıyla, görsellerin LAB renk uzayında işlenip L kanalına **CLAHE (Contrast Limited Adaptive Histogram Equalization)** uygulanması planlanmaktadır.
* **%99 Doğruluk Hedefi:** Domain Shift probleminin çözülmesinin ardından, aydınlatma kalibrasyonu yapılmış veri seti ile Triplet Loss ağının yeniden eğitilmesi ve literatürdeki SOTA (State of the Art) seviyesi olan %99'luk tanımlama eşiğinin yakalanması hedeflenmektedir.
* **Gelişmiş Veritabanı:** JSON tabanlı hiyerarşik dosya sistemi yerine, vektörel aramalarda çok daha hızlı tepki veren Milvus veya Pinecone gibi bir VectorDB entegrasyonu.
