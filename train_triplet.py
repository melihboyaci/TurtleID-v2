import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

class TripletDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, target_size=(224, 224), **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        
        # Sadece içinde birden fazla resim olan kaplumbağa sınıflarını (t001, t004 vb.) al
        self.classes = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d)) and 
                        len(os.listdir(os.path.join(data_dir, d))) > 1]
        
        # Her sınıfın içindeki resim yollarını bir sözlükte (dictionary) topla
        self.image_paths = {c: [os.path.join(data_dir, c, img) for img in os.listdir(os.path.join(data_dir, c))] 
                            for c in self.classes}
        
    def __len__(self):
        # Her epoch'ta gösterilecek batch sayısı — dengeli versiyon: 200 step
        return 200

    def _read_and_preprocess(self, path, augment=False):
        # Resmi oku, boyutlandır ve ResNet50 formatına uygun hale getir
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        # Hafif augmentation: %50 olasılıkla horizontal flip
        # (sağ profil <-> sol profil birbirine dönüşür, model invariance öğrensin)
        if augment and random.random() < 0.5:
            img = cv2.flip(img, 1)

        return preprocess_input(img.astype(np.float32))

    def __getitem__(self, index):
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(self.batch_size):
            # 1. Rastgele bir pozitif sınıf seç (Anchor ve Positive için)
            positive_class = random.choice(self.classes)
            
            # 2. Rastgele bir negatif sınıf seç (Negative için)
            negative_class = random.choice(self.classes)
            while negative_class == positive_class:
                negative_class = random.choice(self.classes)
            
            # 3. Resimleri seç
            anchor_path, positive_path = random.sample(self.image_paths[positive_class], 2)
            negative_path = random.choice(self.image_paths[negative_class])
            
            # 4. Resimleri işle (augmentation açık) ve listeye ekle
            anchors.append(self._read_and_preprocess(anchor_path, augment=True))
            positives.append(self._read_and_preprocess(positive_path, augment=True))
            negatives.append(self._read_and_preprocess(negative_path, augment=True))
            
        return (np.array(anchors), np.array(positives), np.array(negatives)), np.zeros((self.batch_size,))

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

def build_embedding_model(embedding_dim=256):
    # ImageNet ağırlıklarıyla ResNet50'yi yüklüyoruz, son sınıflandırma katmanını atıyoruz
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Modelin ilk katmanlarını "donduruyoruz" (trainable=False). 
    # Sadece son 10 katmanı ve bizim eklediğimiz katmanları eğiteceğiz (Fine-tuning)
    for layer in base_model.layers[:-10]:
        layer.trainable = False
        
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Kendi vektör uzayımızı (256 boyutlu) oluşturuyoruz
    x = layers.Dense(embedding_dim, activation=None)(x)
    
    # KESİN KURALLAR: Cosine Similarity kullanacağımız için L2 Normalization hayati önem taşır!
    # Lambda yerine UnitNormalization katmanı kullanıyoruz (Keras 3 uyumlu, seri hale getirilebilir).
    embeddings = tf.keras.layers.UnitNormalization(axis=1)(x)
    
    return Model(inputs=base_model.input, outputs=embeddings, name="embedding_model")

def build_siamese_model(embedding_model):
    # Ağın 3 girişi olacak: Anchor (Çıpa), Positive (Aynı), Negative (Farklı)
    anchor_input = layers.Input(name="anchor", shape=(224, 224, 3))
    positive_input = layers.Input(name="positive", shape=(224, 224, 3))
    negative_input = layers.Input(name="negative", shape=(224, 224, 3))
    
    # 3 görseli de aynı ağırlıklara sahip TEK BİR modelden geçiriyoruz (Siyam Ağının sırrı budur)
    emb_a = embedding_model(anchor_input)
    emb_p = embedding_model(positive_input)
    emb_n = embedding_model(negative_input)
    
    # Çıktıları (batch, 3, embedding_dim) şeklinde yığınla
    # Lambda yerine Concatenate + Reshape kullanıyoruz (serializable)
    stacked = layers.Concatenate(axis=1)([
        layers.Reshape((1, -1))(emb_a),
        layers.Reshape((1, -1))(emb_p),
        layers.Reshape((1, -1))(emb_n),
    ])
    output = stacked
    
    return Model(inputs=[anchor_input, positive_input, negative_input], outputs=output, name="siamese_network")

class TripletLossModel(Model):
    def __init__(self, siamese_network, margin=0.5, **kwargs):
        super().__init__(**kwargs)
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # Data Generator'dan gelen X = [anchors, positives, negatives]
        X, _ = data
        
        with tf.GradientTape() as tape:
            embeddings = self.siamese_network(X)
            anchor, positive, negative = embeddings[:, 0, :], embeddings[:, 1, :], embeddings[:, 2, :]
            
            # Formül: d(A, P) - d(A, N) + margin
            ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            
            loss = ap_distance - an_distance + self.margin
            loss = tf.maximum(loss, 0.0)
            loss = tf.reduce_mean(loss)
            
        # Geri yayılım (Backpropagation) ile modeli güncelle
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

if __name__ == "__main__":
    # GPU KONTROLÜ
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Bulundu! Model şu an {len(gpus)} adet GPU ile ateşleniyor.")
    else:
        print("⚠️ DİKKAT: GPU bulunamadı, CPU kullanılıyor. Eğitim uzun sürebilir!")
    
    # Dizin yolunu kendi projene göre kontrol et
    DATA_DIR = "data/database" 
    
    print("Veri Yükleyici (Data Generator) hazırlanıyor...")
    train_gen = TripletDataGenerator(DATA_DIR, batch_size=16)
    
    print("Siyam Ağı Mimarisi Kuruluyor...")
    emb_model = build_embedding_model(embedding_dim=256)
    siamese_net = build_siamese_model(emb_model)
    model = TripletLossModel(siamese_network=siamese_net, margin=0.5)
    
    # Öğrenme oranını bilerek çok düşük (0.0001) tutuyoruz ki ResNet50'nin ezberlediği ağırlıklar birden bozulmasın
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    
    print("🚀 Eğitim Başlıyor! (30 epoch, 200 step, horizontal flip augmentation)")

    # Learning rate scheduler: son epoch'larda ince ayar için düşür
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    model.fit(train_gen, epochs=30, callbacks=[lr_scheduler])

    print("✅ Eğitim Bitti! Yeni Beyin (Ağırlıklar) kaydediliyor...")

    # DİKKAT: Bütün Siyam Ağını değil, SADECE Embedding çıkaran kısmı kaydediyoruz.
    # .keras formatı Keras 3 nativ—serialization sorunu yok.
    emb_model.save("turtle_embedding_model.keras")
    print("Kayıt başarılı: turtle_embedding_model.keras")
    print("   Önemli: recognition.py zaten .keras dosyasını yukluyor.")
    print("   embeddings_cache.json dosyasını silmeyi unutma!")