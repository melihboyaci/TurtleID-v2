# TurtleID-v2 Gelişim ve Araştırma Raporu

**Ad:** Melih Boyacı  
**Numara:** 22253073

## Proje Başlığı

**TurtleID-v2: Hiyerarşik Çoklu Ajan Mimarisi ile Deniz Kaplumbağası Kimliklendirme Sistemi**

## Kullanılan Teknolojiler ve Altyapı

Projenin geliştirilmesinde aşağıdaki temel teknolojiler ve altyapı bileşenleri kullanılmıştır:

- **Python 3.10+** — Projenin temel geliştirme dili; tip güvenliği ve modern dil özellikleri için tercih edilmiştir.
- **TensorFlow 2.x & Keras 3** — Triplet Loss tabanlı metrik öğrenme modelinin eğitimi ve çıkarımı için kullanılan derin öğrenme çatısı.
- **OpenCV** — Görüntü okuma, yeniden boyutlandırma, CLAHE uygulaması ve temel görüntü işleme operasyonları için kullanılmıştır.
- **Gemini 2.5 Flash API** — LLM tabanlı ajan doğrulama, semantik görsel analiz ve hata sonrası karar verme mekanizmalarında kullanılan üretken yapay zeka servisi.
- **Streamlit** — Kullanıcı arayüzü geliştirmede kullanılan Python tabanlı web uygulama çatısı; Human-in-the-Loop etkileşimi bu arayüz üzerinden sağlanmaktadır.
- **WSL2 & CUDA** — GPU ivmelendirmeli model eğitimi için kullanılan Linux alt sistemi ve NVIDIA paralel hesaplama platformu.

## Raporun Amacı

Bu rapor, TurtleID-v2 projesinin yedi günlük Ar-Ge geliştirme sürecini günlük formatında belgelemek amacıyla hazırlanmıştır. Çalışmanın temel hedefi, deniz kaplumbağalarının kafa profili görüntülerinden bireysel kimlik tespiti yapabilen; hata durumlarında güvenli karar verebilen; modüler, sürdürülebilir ve genişletilebilir bir yazılım mimarisi geliştirmektir.

Proje sürecinde ilk denemeler klasik görüntü işleme ve derin öğrenme yaklaşımlarıyla başlamış, ancak zamanla sistem hiyerarşik bir çoklu ajan mimarisine dönüştürülmüştür. Son aşamada ise görüntü işleme algoritmalarının biyolojik görüntülerdeki karmaşık dokulara karşı yanılabildiği görülmüş ve sistem **Human-in-the-Loop** yaklaşımıyla daha güvenilir hale getirilmiştir.

---

## 1. Gün: Araştırma ve Spagetti Kod Çıkmazı

İlk gün çalışmanın odağı, deniz kaplumbağası kimliklendirme probleminin teknik olarak nasıl çözülebileceğini araştırmaktı. Literatürde birey tanıma problemlerinde özellikle yüz, kabuk, leke, damar ve doku örüntülerinin ayırt edici özellikler taşıdığı görüldü. Deniz kaplumbağaları özelinde ise kafa profili üzerindeki pulların ve yüzey desenlerinin bireyler arasında ayırt edici olabileceği değerlendirildi.

Bu doğrultuda ilk prototipte iki temel teknoloji kullanıldı:

- **OpenCV:** Görüntü okuma, yeniden boyutlandırma, kırpma ve basit kontur analizi için kullanıldı.
- **ResNet50:** Önceden eğitilmiş derin öğrenme modeli olarak görsel özellik çıkarımı için test edildi.

İlk denemelerde tüm işlemler tek bir dosya içinde, sıralı ve monolitik biçimde yazıldı. Görselin okunması, kafa bölgesinin bulunması, görüntünün modele hazırlanması, embedding çıkarılması, veritabanı ile karşılaştırılması ve sonuç üretimi aynı akış içinde yürütülüyordu. Bu yaklaşım kısa vadede hızlı prototipleme sağladı; ancak birkaç testten sonra ciddi mimari problemler ortaya çıktı.

Karşılaşılan başlıca sorunlar şunlardı:

- **Hata izolasyonu yoktu:** Görsel okunamadığında mı, kafa bulunamadığında mı, model embedding üretemediğinde mi hata oluştuğu net biçimde ayrıştırılamıyordu.
- **Kod genişletilemiyordu:** Yeni bir doğrulama adımı eklemek veya mevcut algoritmayı değiştirmek tüm akışı etkileme riski taşıyordu.
- **Test edilebilirlik düşüktü:** Tek parça kod nedeniyle her modül bağımsız olarak sınanamıyordu.
- **Bakım maliyeti artıyordu:** Küçük bir değişiklik bile beklenmeyen yan etkiler oluşturabiliyordu.

Bu aşamada projenin yalnızca bir görüntü işleme problemi olmadığı, aynı zamanda doğru bir yazılım mimarisi problemi olduğu anlaşıldı. Monolitik yapı, araştırma projesi için yeterli görünse de akademik değerlendirme, sürdürülebilirlik ve hata yönetimi açısından zayıf kalıyordu.

Günün sonunda alınan temel karar, prototipin yeniden yapılandırılması ve sorumlulukların bağımsız bileşenlere ayrılması oldu.

---

## 2. Gün: Çoklu Ajan Mimarisine Geçiş

İkinci gün, sistemin laboratuvar derslerinde ele alınan hiyerarşik çoklu ajan yaklaşımına uygun biçimde yeniden tasarlanmasına ayrıldı. Bu aşamada amaç, tek parça çalışan kodu görev paylaşımı yapan ajanlardan oluşan bir sisteme dönüştürmekti.

Yeni mimaride merkezde bir yönetici ajan konumlandırıldı:

- **SupervisorAgent:** Tüm görevi yöneten, hangi ajanın ne zaman çalışacağını belirleyen ve hata durumlarında karar veren üst düzey koordinatör ajan.

`SupervisorAgent`, doğrudan görüntü işleme veya tanıma işlemini yapmak yerine bu işleri uzmanlaşmış worker ajanlara delege edecek şekilde tasarlandı. Böylece sistemdeki her ajan yalnızca kendi sorumluluk alanına odaklandı.

Bu gün geliştirilen ikinci kritik yapı ise paylaşılan hafıza katmanı oldu:

- **BlackBoard:** Ajanların ortak olarak okuyup yazabildiği merkezi durum deposu.

`BlackBoard`, sistemin görev girdisini, ara çıktıları, hata mesajlarını, görev durumunu ve ajan loglarını tutan ortak bir bellek olarak kurgulandı. Örneğin `AuditWorker` görselin okunabilir olduğunu doğruladığında sonucu `BlackBoard` üzerine yazmakta; `RecognitionWorker` embedding vektörlerini buradan okumakta; `EvaluationWorker` ise eşleşme sonucunu yine aynı yapı üzerinden sisteme bildirmektedir.

Bu yaklaşımla proje aşağıdaki hiyerarşik akışa kavuştu:

1. `SupervisorAgent` görevi başlatır.
2. Sıradaki worker ajanı çalıştırır.
3. Worker ajan sonucu `BlackBoard` üzerine yazar.
4. `SupervisorAgent` sonucu değerlendirir.
5. Başarı durumunda bir sonraki adıma geçilir.
6. Hata durumunda sistem güvenli karar mekanizmasına yönelir.

Bu mimari geçiş, projenin yalnızca çalışan bir prototip olmaktan çıkıp izlenebilir, açıklanabilir ve yönetilebilir bir çoklu ajan sistemine dönüşmesini sağladı.

---

## 3. Gün: SOLID Prensipleri ve İzolasyon

Üçüncü günün ana hedefi, geliştirilen çoklu ajan yapısını Clean Code ve SOLID prensipleriyle uyumlu hale getirmekti. İlk iki günde ajanlara ayrılan yapı işlevsel olsa da, her ajanın ortak bir davranış sözleşmesine sahip olması gerekiyordu.

Bu nedenle tüm worker ajanların miras aldığı soyut bir temel sınıf geliştirildi:

- **BaseWorker:** Worker ajanlar için ortak arayüz sağlayan soyut temel sınıf.

`BaseWorker` yapısı ile her worker ajan için ortak kurallar belirlendi:

- Her ajan bir `BlackBoard` referansı ile çalışır.
- Her ajan `execute()` metodunu uygulamak zorundadır.
- Her ajan başarı veya başarısızlık bilgisini standart biçimde döndürür.
- Her ajan kendi adını ve loglarını ortak formatta sisteme aktarır.

Bu yapı özellikle SOLID prensipleri açısından önemli kazanımlar sağladı.

### Single Responsibility Principle

Her ajan tek bir göreve odaklandı:

- **AuditWorker:** Girdi dosyasının okunabilirliğini ve temel geçerliliğini kontrol eder.
- **HeadDetectionWorker:** Görselin deniz kaplumbağası yan kafa profili olup olmadığını doğrular.
- **PreprocessingWorker:** Görseli modelin beklediği forma getirir.
- **RecognitionWorker:** ResNet50 ile embedding üretir.
- **EvaluationWorker:** Kosinüs benzerliği ile eşleşme kararını verir.
- **ReportingWorker:** Sonucu yorumlayıp rapor üretir.

Bu ayrım sayesinde her sınıfın değişme nedeni sınırlandırılmış oldu.

### Open/Closed Principle

Sistem değişime kapalı, gelişime açık hale getirildi. Yeni bir worker ajan eklemek için mevcut ajanların iç yapısını bozmak gerekmemektedir. Yeni ajan `BaseWorker` sözleşmesine uyarak sisteme dahil edilebilir ve `SupervisorAgent` pipeline sırasına eklenebilir.

Örneğin ileride `QualityAssessmentWorker` veya `SpeciesValidationWorker` gibi yeni ajanlar eklenmek istenirse, mevcut `RecognitionWorker` ya da `EvaluationWorker` sınıflarını değiştirmeden genişleme yapılabilir.

### Dependency Inversion Principle

`SupervisorAgent`, worker ajanlarla ortak bir davranış modeli üzerinden çalışır. Ajanların her biri `execute()` metoduna sahip olduğundan, yönetici ajanın tüm iç algoritmaları bilmesi gerekmez. Bu durum üst seviye kontrol mantığının alt seviye işlem detaylarına bağımlılığını azaltmıştır.

Bu günün sonunda proje, spagetti koddan çıkarak okunabilir, genişletilebilir ve sınanabilir bir yazılım mimarisine kavuştu.

![Hiyerarşik Çoklu Ajan (MAS) ve BlackBoard Akış Diyagramı](turtle-id/assets/architecture_diagram.png)

*Şema 1: SupervisorAgent, BlackBoard ve Worker ajanlar arasındaki asenkron veri akışı ve görev delegasyonu.*

---

## 4. Gün: Yapay Zeka Entegrasyonu ve Hibrit Sistem

Dördüncü gün, sistemin yalnızca deterministik algoritmalardan oluşmasının yeterli olmadığı görüldü. OpenCV ve ResNet50 güçlü araçlar olsa da, bazı kararların yalnızca piksel seviyesinde değil anlamsal düzeyde değerlendirilmesi gerekiyordu.

Bu nedenle sisteme LLM destekli yapay zeka katmanı eklendi:

- **Gemini 2.5 Flash:** Görsel doğrulama, hata değerlendirme ve raporlama süreçlerinde kullanılan üretken yapay zeka modeli.

Bu entegrasyonla sistem hibrit bir mimariye dönüştü. Deterministik ajanlar sayısal işlemleri ve model tabanlı hesaplamaları yürütürken, LLM destekli bileşenler daha yorumlayıcı karar noktalarında görev aldı.

Bu gün öne çıkan üç entegrasyon noktası şunlardı:

### 1. Görsel Doğrulama

`HeadDetectionWorker`, Gemini Vision desteğiyle kullanıcının verdiği görselin gerçekten bir deniz kaplumbağası yan kafa profili olup olmadığını kontrol etmeye başladı. Böylece sistem yalnızca görüntünün teknik olarak okunabilir olmasına değil, biyolojik ve semantik olarak doğru girdi olmasına da dikkat etmeye başladı.

### 2. Hata Sonrası Karar Verme

`SupervisorAgent`, bir worker başarısız olduğunda Gemini'ye danışarak hatanın kurtarılabilir olup olmadığını değerlendirecek şekilde tasarlandı. Bu karar mekanizması, sistemin her hatada kör biçimde devam etmesini engelledi.

Özellikle kimliklendirme gibi hassas bir görevde yanlış veriyle devam etmek, yanlış birey eşleştirmesine neden olabilir. Bu nedenle hata durumunda güvenli duruş stratejisi kritik hale geldi.

### 3. Raporlama

`ReportingWorker`, görev sonucunu ve görev loglarını Gemini ile yorumlayarak gelişim raporu üretebilen bir ajan olarak sisteme eklendi. Bu sayede sistem yalnızca sonuç veren değil, kendi sürecini açıklayabilen bir yapıya yaklaşmıştır.

Bu günün sonunda TurtleID-v2, klasik görüntü işleme hattından çıkarak deterministik işlem adımları ile LLM destekli yorumlama katmanını birleştiren hibrit bir çoklu ajan sistemine dönüştü.

---

## 5. Gün: Metrik Öğrenme, GPU İvmelendirmesi ve Embedding Altyapısı

Beşinci gün, projenin en kritik teknik dönüşümüne sahne oldu. ResNet50 modeliyle yapılan ilk embedding testlerinde farklı bireyler arasında benzerlik skorlarının `%87` ile `%96` bandında kümelenebildiği gözlemlendi. Bu durum, genel amaçlı ImageNet ön eğitiminin bireysel kimlik tespiti için yeterli olmadığını açıkça ortaya koydu.

Kök neden analizi şu bulguya dayandı: ImageNet, deniz kaplumbağalarını bireysel kimlik düzeyinde değil `sea turtle` genel sınıfı düzeyinde temsil eder. Dolayısıyla ResNet50'nin öğrendiği embedding uzayı türler arası ayrım için optimize edilmiş; ancak bireyler arası post-oküler pul deseni farklarını yakalamak için gerekli **metrik öğrenme** kapasitesinden yoksundu.

Bu bulgular doğrultusunda `RecognitionWorker` içindeki özellik çıkarım mekanizmasının alana özgü bir modelle değiştirilmesi kararlaştırıldı. Seçilen yaklaşım **Triplet Loss** tabanlı metrik öğrenmeydi. Bu yöntemde model, aynı bireye ait iki görüntüyü (anchor–positive) embedding uzayında birbirine yaklaştırırken, farklı bireye ait görüntüyü (negative) uzaklaştıracak biçimde eğitilir.

### GPU İvmelendirmesi — WSL2 Ortamına Geçiş

438 bireylik veri setiyle Triplet Loss eğitimi başlatıldığında ciddi bir performans duvarına çarpıldı. Windows ortamında TensorFlow'un GPU desteğinin kısıtlı çalışması nedeniyle ilk denemeler CPU'ya düşmekte, epoch başına süre kabul edilemez boyutlara ulaşmaktaydı.

Sorunu aşmak için eğitim ortamı **WSL2 (Windows Subsystem for Linux 2)** üzerine taşındı. Linux ortamında TensorFlow'un CUDA desteği tam olarak devreye girdi ve GPU ivmelendirmesi aktif hale geldi. Bu geçişin etkisi çarpıcıydı: **2 saati aşan eğitim süresi 35 dakikaya düştü.** Bu iyileştirme yalnızca geliştirme hızını değil, hiper-parametre denemeleri ve yeniden eğitim döngülerinin sürdürülebilirliğini de doğrudan etkiledi.

### Embedding Altyapısı ve Cache Sistemi

Eğitim tamamlandıktan sonra `t001`'den `t500`'e kadar numaralandırılmış **438 kaplumbağa bireyi** için embedding vektörlerinin üretilmesi gerekiyordu. Her bireyin `head_left` ve `head_right` gibi birden fazla profil görseli bulunduğundan, her çalıştırmada bu görselleri yeniden modelden geçirmek maliyetliydi.

Bu problemi çözmek için bir **Embedding Cache Sistemi** tasarlandı. Sistem, her kaplumbağa klasörü için üretilen embedding listesini ve klasördeki görsel sayısını saklıyor; görsel sayısı değişmediği sürece `RecognitionWorker` cache'i kullanıyor, yeni görsel eklenmesi durumunda ilgili bireyin kaydını otomatik olarak yeniliyordu.

İlk tasarımda birden fazla profile ait embedding'lerin **ortalaması** alınarak bireyin temsil vektörü oluşturuluyordu. Bu yaklaşım mantıklı görünse de sonraki testlerde önemli bir zayıflığı ortaya çıkaracaktı (bkz. 6. Gün).

`EvaluationWorker`, üretilen embedding'leri kosinüs benzerliği üzerinden karşılaştırarak eşleşme kararı verdi. Sistem üç karar kategorisine ayrıldı:

- **GÜÇLÜ_EŞLEŞME:** Benzerlik skoru `0.85` ve üzerinde.
- **OLASI_EŞLEŞME:** Benzerlik skoru `0.70` ile `0.85` arasında.
- **YENİ_BİREY:** Benzerlik skoru `0.70` altında.

Bu günün sonunda alana özgü bir Triplet Loss modeli eğitilmiş, GPU ivmelendirmeli bir geliştirme altyapısı kurulmuş ve 438 bireylik veritabanı için embedding cache mekanizması çalışır hale getirilmişti. Artık sistemin bilimsel çekirdeği gerçek anlamda biyometrik kimliklendirme için tasarlanmış bir modele dayanıyordu.

---

## 6. Gün: Kriz Yönetimi — İki Kritik Hata ve Mimari Savunma

Altıncı gün, projenin en yoğun hata ayıklama dönemiydi. Özel Triplet Loss modeliyle bile bazı test sorgularında yanlış pozitif sonuçlar devam ediyordu. `t007` bireyine ait sorgu görselinin `t084` ile eşleşmesi en dikkat çekici örneklerden biriydi. Arka arkaya iki ayrı kriz analiz edildi ve her biri mimari prensiplerden ödün verilmeden çözüme kavuşturuldu.

### Kriz 1: Eğitim-Çıkarım (Training-Inference) Domain Shift

Pipeline denetimi yapıldığında, eğitim ve çıkarım aşamaları arasında kritik bir veri dönüşümü uyuşmazlığı tespit edildi:

- **Eğitim hattı:** `Ham RGB → Yeniden Boyutlandırma → Preprocess → Model`
- **Çıkarım hattı:** `Ham RGB → CLAHE → Yeniden Boyutlandırma → Preprocess → Model`

5. günde deneysel olarak eklenen CLAHE görüntü iyileştirmesi yalnızca çıkarım hattına eklenmiş; ancak model bu dönüşüm uygulanmış görüntülerle hiç eğitilmemişti. Triplet Loss ile eğitilen model ham RGB dağılımından gelen görseller üzerinde embedding uzayı oluştururken, çıkarım sırasında lokal kontrastı artırılmış ve piksel dağılımı değişmiş görüntüler modele veriliyordu. Kosinüs benzerliği metriği teknik olarak doğru çalışmasına rağmen, karşılaştırılan vektörler modelin öğrendiği dağılımdan saptığı için skorlar güvenilirliğini kaybetmişti.

Çözüm, makine öğrenmesinin temel prensibine dayandı:

**Eğitim ve çıkarım hatları birebir aynı veri dönüşümlerini kullanmalıdır.**

Bu doğrultuda `RecognitionWorker` ajanının `_extract()` metodundan CLAHE kaldırıldı; sistem ham `head_crop` görselleri üzerinden embedding üretmeye döndürüldü. Bozulmuş cache dosyası geçersiz sayıldı ve tüm veritabanı embedding'leri sıfırdan yeniden üretildi.

### Mimari Savunma: Single Responsibility Principle Korundu

Bu noktada kritik bir tasarım sorusu gündeme geldi: CLAHE kaldırıldığına göre `PreprocessingWorker` ajanının rolü ne olacaktı? Ajanı pipeline'dan çıkarmak kolay bir çözüm gibi görünse de bu yaklaşım, 3. günde temelleri atılan **Single Responsibility Principle**'ı ihlal ederdi; her ajan tek bir sorumluluğa sahip olmalı ve o sorumluluk değiştiğinde ajan silinmemeli, güncellenmeli.

Bunun yerine `PreprocessingWorker`'ın sorumluluk tanımı yeniden yazıldı:

- **Eski görev:** Görüntü iyileştirme (CLAHE, Unsharp Masking vb.)
- **Yeni görev:** **Model için Tensör Hazırlığı** — doğrulanmış kafa görselini `224×224` boyutuna yeniden boyutlandırıp `preprocess_input()` ile modelin beklediği tensör formatına dönüştürmek.

Ajanın sorumluluğu daraltıldı ve netleştirildi; ancak pipeline'dan çıkarılmadı. **6-ajanlı yapı ve Tek Sorumluluk Prensibi kusursuz biçimde korundu.**

### Kriz 2: Averaging Blur — Vektör Ortalamasının Gizli Zayıflığı

Domain shift krizi çözüldükten sonra bazı yanlış pozitifler hâlâ devam ediyordu. İkinci analizde sorunun farklı bir yerden kaynaklandığı anlaşıldı: **vektör ortalaması.**

Birden fazla görüntüye sahip bir kaplumbağa için tüm embedding'lerin ortalaması alındığında, embedding uzayındaki keskin bireysel özellikler yumuşatılmaktaydı. Örneğin beş farklı açıdan çekilmiş beş görüntünün embedding ortalaması, o bireye ait hiçbir gerçek görüntünün temsiline tam olarak karşılık gelmeyen yapay bir orta nokta oluşturuyordu. Bu fenomen **Averaging Blur** olarak adlandırıldı.

Çözüm olarak **Max-of-Images (Galeri/Probe)** mimarisine geçildi. Bu yaklaşımda ortalama alınmaz; bunun yerine sorgu embedding'i veritabanındaki bireye ait tüm ayrı embedding'lerle birer birer karşılaştırılır ve bu karşılaştırmalar arasındaki en yüksek skor o bireyin skoru olarak kabul edilir. Böylece bireyin veritabanındaki en iyi görüntüsüyle eşleşme şansı korunmuş olur.

Bu değişiklik için cache yapısı tamamen güncellendi. Artık her birey için tek bir ortalama vektör değil, tüm profil görüntülerine ait ayrı embedding listesi saklanıyordu. Cache bu yeni formatla sıfırdan yeniden oluşturuldu.

### Güvenli Duruş Testi

Kriz çözümlerinin ardından sistemin hatalı girdilere karşı savunma katmanı da sınandı. Sisteme kafa olmayan görüntüler — gövde ağırlıklı, kabuk, yüzgeç veya tamamen alakasız fotoğraflar — verildi. `HeadDetectionWorker` bu girdileri Gemini destekli doğrulama ile erken aşamada reddetti; `SupervisorAgent` hata kararını `BlackBoard` üzerinden okuyarak pipeline'ı güvenli şekilde durdurdu. Sistem **fail-safe** davranışını da başarıyla sergiledi.

Altıncı günün sonunda iki kriz de çözülmüş, pipeline temizlenmiş, SRP korunmuş ve Max-of-Images mimarisine geçilmişti. Sistem, final doğruluk ölçümü için hazırdı.

---

## 7. Gün: Human-in-the-Loop, Gemini Kalibrasyonu ve Final Zaferi

Yedinci ve final gününde üç kritik konu ele alındı: girdi güvenilirliğini sağlayan mimari kararın olgunlaştırılması, `HeadDetectionWorker`'ın doğrulama davranışının optimize edilmesi ve tüm iyileştirmelerin birleşik etkisinin ölçülmesi.

### Human-in-the-Loop Kararı

Önceki aşamalarda `HeadDetectionWorker` için OpenCV tabanlı otomatik kafa bulma denemeleri yapılmıştı. Ancak gerçek deniz kaplumbağası görüntülerinde otomatik tespit yaklaşımının beklenenden daha problemli olduğu görüldü. Temel sorun, deniz kaplumbağalarının kabuk, yüzgeç ve deri dokularının yüksek kenar yoğunluğuna sahip olmasıydı. OpenCV tabanlı algoritmalar, kenar ve kontur yoğunluğunu kafa bölgesi sanabiliyor; biyolojik olarak zengin dokuları yanlışlıkla aday bölge olarak seçebiliyordu. Yanlış kırpılan görsel embedding üretim hattına girdiğinde, doğruluk kaybının kaynağını tespit etmek giderek zorlaşıyordu.

Bu gözlem doğrultusunda final mimari kararı alındı:

**TurtleID-v2, Human-in-the-Loop yaklaşımına geçirildi.**

Kafa bölgesinin manuel kırpılması kullanıcı sorumluluğuna alındı; böylece en riskli adım insan uzmanlığıyla desteklendi. Bu değişiklikle `HeadDetectionWorker` ajanının rolü de yeniden tanımlandı:

- **HeadDetectionWorker:** Kullanıcının verdiği manuel kırpılmış görselin gerçekten deniz kaplumbağası yan kafa profili olup olmadığını Gemini Vision ile doğrulayan validator ajan.

### Gemini Doğrulama Kriterlerinin Kalibrasyonu

`HeadDetectionWorker` doğrulama kriterleri başlangıçta oldukça katıydı. Sahadaki fotoğrafların gerçekçi koşullarla sınanması sırasında, Gemini'nin geçerli kafa fotoğraflarını bile reddetebildiği gözlemlendi. Bulanık, az aydınlık veya kısmen çerçeve dışında kalmış ama biyolojik olarak geçerli profil görüntüleri, aşırı kısıtlayıcı prompt kuralları nedeniyle reddediliyordu.

Bu nedenle doğrulama kriterleri biyolojik gerçekçilik açısından esnetildi. Prompt, "görüntüde bir kaplumbağa kafa profili seçilebiliyorsa kabul et" yönünde yeniden yapılandırıldı; yalnızca kafa olmayan (kabuk, gövde, yüzgeç veya tamamen alakasız) görseller reddedilecek biçimde kural setleri netleştirildi. Bu kalibrasyon, sistemin doğru girdi üzerinde **false-reject (yanlış red)** yapmasını önlerken hatalı girdilere karşı fail-safe davranışını korudu.

### Girdi Kalitesi Kontrolü ve CLAHE Entegrasyonu (Geri Dönüş)

Saha testlerinin ilerleyen aşamalarında, "t205" gibi uzaktan çekilmiş, kafa alanının çok küçük ve bulanık (örneğin 60x60 piksel altı) olduğu görsellerde modelin zorlandığı tespit edildi. Bu problem iki temel ML kuralına dayanıyordu: "Garbage in, garbage out" (Çöp girerse, çöp çıkar) ve düşük çözünürlüklü görsellerin modelin beklediği 224x224 boyutuna çekiştirildiğinde (upscaling) tamamen çamurlaşması.

Bu krizi aşmak için iki stratejik hamle yapıldı:

1. **Arayüz Kalite Kontrolü (Strict Mode):** Kullanıcının arayüzde seçebileceği minimum kırpma alanı (`MIN_SIZE`) 100x100 piksele çıkarıldı. Böylece sisteme bilgi taşımayan, aşırı pikselli görsellerin girmesi baştan engellendi.
2. **Tam Uyumlu CLAHE Entegrasyonu:** 6. günde "Eğitim-Çıkarım uyuşmazlığı" (Domain Shift) nedeniyle iptal edilen CLAHE (Contrast Limited Adaptive Histogram Equalization) filtresi projeye geri döndürüldü. Ancak bu kez hata tekrarlanmadı; CLAHE hem modelin eğitim koduna (`train_triplet.py`) hem de çıkarım hattına (`agents/tensor_utils.py`) simetrik biçimde yerleştirildi. Bu sayede ResNet50 modeli, doğrudan su altı fotoğraflarındaki vurgulanmış, keskinleştirilmiş pul dokularını görerek eğitildi.

### Final Doğruluk Ölçümü

Tüm iyileştirmeler — Triplet Loss modeli, WSL2/GPU eğitimi, domain shift düzeltmesi, Max-of-Images mimarisi, cache yenileme ve Gemini kalibrasyonu — entegre edildikten sonra kapsamlı testler gerçekleştirildi. Sistem, 438 bireye ait toplam 8526 adet görüntülük veritabanı üzerinde test edilmiş olup, her birey için birden fazla profil açısını içeren galeri görüntüleri kullanılmıştır. Test metodolojisi, veritabanında mevcut bireylerin farklı açılardan çekilmiş sorgu görüntüleriyle eşleştirilmesi ve sonuçların kosinüs benzerliği eşik değerlerine göre değerlendirilmesi esasına dayanmaktadır.

Sonuçlar, projenin başlangıcındaki hedeflerin çok ötesine geçildiğini gösterdi:

- **T003:** `%97.5` benzerlik skoru ile doğru birey eşleşmesi.
- **T007:** `%92.8` benzerlik skoru ile doğru birey eşleşmesi.
- **Genel sistem doğruluğu:** `%90` üzeri.

Başlangıçta belirlenen `%60` doğruluk hedefi büyük ölçüde aşıldı. Bu başarı, tek bir bileşenin değil; mimari kararlar, kriz yönetimi ve algoritma optimizasyonlarının birleşik etkisinin sonucudur.

Final pipeline akışı şu hale geldi:

1. İnsan kullanıcı kafa bölgesini manuel kırpar.
2. `AuditWorker` dosyanın teknik olarak geçerli olup olmadığını kontrol eder.
3. `HeadDetectionWorker` görselin semantik olarak doğru kafa profili olup olmadığını Gemini ile doğrular.
4. `PreprocessingWorker` doğrulanmış görseli `224×224` boyutuna yeniden boyutlandırıp modelin beklediği tensör formatına dönüştürür.
5. `RecognitionWorker` özel Triplet Loss modeli ile embedding vektörü üretir.
6. `EvaluationWorker` Max-of-Images yaklaşımıyla kosinüs benzerliği hesaplar ve eşleşme kararı verir.
7. `ReportingWorker` sonucu ve görev loglarını raporlar.
8. `SupervisorAgent` tüm akışı yönetir ve hata durumlarında güvenli duruşu sağlar.

TurtleID-v2'nin final tasarımı, otomasyon ile insan denetimini dengede tutan, hata durumlarında güvenli duran ve bireysel kimlik tespitinde `%90+` doğruluğa ulaşan akademik ve pratik açıdan savunulabilir bir sisteme dönüşmüştür.

---

## Final Mimari Değerlendirme

Yedi günlük geliştirme süreci sonunda TurtleID-v2, monolitik bir görüntü işleme denemesinden hiyerarşik, hibrit ve insan denetimli bir çoklu ajan sistemine dönüşmüştür.

Final sistemde ajanların görevleri net biçimde ayrılmıştır:

| Ajan                  | Görev                                                                                                                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SupervisorAgent`     | Tüm pipeline akışını yönetir, görevleri worker ajanlara delege eder, hata durumunda Gemini'ye danışarak güvenli karar verir.                                                                 |
| `AuditWorker`         | Girdi dosyasının format, okunabilirlik ve temel geçerlilik kontrollerini yapar.                                                                                                              |
| `HeadDetectionWorker` | Manuel kırpılmış görselin deniz kaplumbağası yan kafa profili olup olmadığını Gemini Vision ile doğrulayan validator ajan.                                                                   |
| `PreprocessingWorker` | Doğrulanmış kafa görselini `224×224` boyutuna yeniden boyutlandırıp `preprocess_input()` ile modelin beklediği tensör formatına dönüştürür.                                                  |
| `RecognitionWorker`   | Triplet Loss ile eğitilmiş özel model üzerinden embedding vektörü üretir; Max-of-Images yaklaşımıyla tüm profil görüntülerinin ayrı embedding'lerini yönetir ve cache mekanizmasını işletir. |
| `EvaluationWorker`    | Max-of-Images kosinüs benzerliği ile en yakın bireyi bulur ve eşleşme statüsünü belirler.                                                                                                    |
| `ReportingWorker`     | Görev sonucunu ve görev loglarını Gemini ile yorumlayarak rapor üretir.                                                                                                                      |
| `BlackBoard`          | Tüm ajanların ortak hafızası olarak ara çıktıları, hata mesajlarını ve görev loglarını tutar.                                                                                                |

Yazılım mimarisi açısından en önemli kazanım, `BaseWorker` soyut sınıfı ile ajanlar arasında ortak bir sözleşme kurulmasıdır. Bu yapı sayesinde sistem Clean Code ilkelerine uygun hale gelmiş; sorumluluklar ayrılmış, bağımlılıklar azaltılmış ve kriz anlarında değişiklikler izole edilmiştir. 6. gündeki `PreprocessingWorker` örneğinde görüldüğü üzere bir ajanın sorumluluğu değiştiğinde sistem yeniden yazılmamış; yalnızca ilgili ajanın görev tanımı güncellenmiştir.

Sonuç olarak TurtleID-v2, yalnızca bir tanıma algoritması değil; hata yönetimi, kriz izolasyonu, açıklanabilirlik, modülerlik ve insan denetimini birlikte ele alan; `%90+` doğruluk oranına ulaşmış bütüncül bir yapay zeka sistemi olarak konumlandırılmıştır.

---

## Proje Kriz Geçmişi, Aşılan Sınırlamalar ve Gelecek Vizyonu

Yedi günlük geliştirme süreci boyunca sistem üç kritik sınırlamayla karşılaştı. Bu sınırlamaların her biri tanımlanmış, kök nedeni bulunmuş ve SOLID mimarisi ödün verilmeden aşılmıştır.

### Aşılan Sınırlama 1: ImageNet Ön Eğitimi Yetersizliği

İlk testlerde genel amaçlı ResNet50 modeli, farklı bireyler arasında benzerlik skorlarının `%87`–`%96` bandında kümelenmesine yol açıyordu. Kök neden, modelin ImageNet üzerinde bireysel kimlik değil genel sınıf ayrımı için eğitilmiş olmasıydı. Çözüm olarak 438 bireylik veritabanı kullanılarak **Triplet Loss** tabanlı özel bir model eğitildi; model artık "bu iki profil aynı bireye mi ait?" sorusunu metrik öğrenme yöntemiyle öğrendi.

### Aşılan Sınırlama 2: Eğitim-Çıkarım Domain Shift

Triplet Loss modeli devreye alındıktan sonra bile yanlış pozitifler devam etti. Pipeline denetiminde eğitim hattı (`Ham RGB → Model`) ile çıkarım hattı (`Ham RGB → CLAHE → Model`) arasındaki uyuşmazlık tespit edildi. `RecognitionWorker`'ın `_extract()` metodundan CLAHE kaldırıldı; sistem eğitimle tutarlı ham görüntüler üzerinden çalışmaya döndürüldü. Bu süreçte `PreprocessingWorker` ajanı silinmedi; görevi "görüntü iyileştirme"den "Model için Tensör Hazırlığı (`224×224` + `preprocess_input()`)" olarak güncellendi. **6-ajanlı yapı ve Single Responsibility Principle korundu.**

### Aşılan Sınırlama 3: Averaging Blur

Birden fazla görüntüye sahip bireylerde embedding vektörlerinin ortalaması alındığında, bireysel ayırt edici özellikler yumuşatılıyordu. Çözüm olarak **Max-of-Images (Galeri/Probe)** mimarisine geçildi; her bireyin tüm profil görüntüleri ayrı ayrı karşılaştırıldı ve en yüksek skor o bireyin skoru kabul edildi.

### Mimarinin Kanıtlanmış Gücü

Bu üç krizin ortak paydası, her çözümün yalnızca `RecognitionWorker` ajanı içinde izole edilmesidir. `SupervisorAgent`, `AuditWorker`, `HeadDetectionWorker`, `EvaluationWorker` ve `ReportingWorker` ajanlarına hiç dokunulmadı. Bu durum, `BaseWorker` soyutlamasının ve Open/Closed Principle'ın pratikte ne anlama geldiğini doğrulayan somut bir kanıttır.

### Mevcut Durum ve Gelecek Vizyonu

Tüm krizler aşıldıktan sonra sistem T003'te `%97.5`, T007'de `%92.8` ve genel olarak `%90+` doğruluk oranına ulaştı. Başlangıç hedefi olan `%60`'ın çok üzerindeki bu sonuç, mimarinin, algoritmanın ve kriz yönetiminin birleşik etkisini yansıtmaktadır.

Gelecek geliştirme ekseninde iki temel konu öne çıkmaktadır. Birincisi, Triplet Loss modelin veri seti genişletilerek ve daha uzun eğitim süreleriyle iyileştirilmesidir. İkincisi, `RecognitionWorker` içine **Siamese Network** tabanı eklenerek çift görüntü karşılaştırma kapasitesinin artırılmasıdır. Bu geliştirmelerin hiçbiri diğer ajanlara dokunmayı gerektirmez; sistem Open/Closed Principle sayesinde bu güncellemelere hazır durumdadır.
