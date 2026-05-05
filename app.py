"""
app.py — TurtleID Ar-Ge ve Sunum Arayüzü (Streamlit)
=====================================================

Bu arayüz, TurtleID Çoklu Ajan (MAS) sistemini jüriye/hocaya sunmak için
geliştirilmiş modern ve şık bir ön yüzdür. Arka plandaki mevcut
ajan mimarisini bozmadan süreci görselleştirir.

Kullanım:
    streamlit run app.py
"""

import os
import streamlit as st
from PIL import Image

from agents.supervisor import SupervisorAgent
from config import QUERY_DIR, DATABASE_DIR

# ---------------------------------------------------------
# Sayfa Yapılandırması ve Premium Tasarım (CSS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="TurtleID Ar-Ge Arayüzü",
    page_icon="🐢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2E7D32;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Yan Menü (Sidebar) - Sistem Metrikleri ve Ajan Mimarisi
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Sistem Metrikleri")
    st.metric(label="Kullanılan Model", value="turtle_embedding_model")
    st.metric(label="Eşleştirme Algoritması", value="Max-of-Images Kosinüs")
    
    st.markdown("---")
    st.markdown("### 🤖 Ajan Mimarisi (MAS)")
    st.markdown("- 👑 **SupervisorAgent**: Süreç Yöneticisi")
    st.markdown("- 🔍 **AuditWorker**: Girdi Doğrulama")
    st.markdown("- 🧠 **HeadDetectionWorker**: Kafa Tespiti (Gemini Vision)")
    st.markdown("- 🛠️ **PreprocessingWorker**: Tensör İşlemleri")
    st.markdown("- 🧬 **RecognitionWorker**: Embedding Çıkarımı")
    st.markdown("- ⚖️ **EvaluationWorker**: Benzerlik Puanlama")
    st.markdown("- 📝 **ReportingWorker**: Raporlama")

# ---------------------------------------------------------
# Ana Ekran
# ---------------------------------------------------------
st.markdown('<p class="main-title">🐢 TurtleID Çoklu Ajan Sistemi</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ar-Ge Sunum ve Canlı Test Arayüzü</p>', unsafe_allow_html=True)

st.write("### 📸 Sorgu Görseli Yükle")
uploaded_file = st.file_uploader(
    "Kaplumbağa yan profil (kafa) fotoğrafı yükleyin (JPG, PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col_input1, col_input2 = st.columns([1, 2])
    
    with col_input1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)
    
    with col_input2:
        st.info("✅ Görsel sisteme başarıyla yüklendi. Otonom ajan mimarisini başlatmak için aşağıdaki butona tıklayabilirsiniz.")
        start_button = st.button("🚀 MAS Pipeline'ı Başlat", type="primary", use_container_width=True)

    # Butona Tıklandığında Pipeline Başlar
    if start_button:
        st.markdown("---")
        
        # 1. Yüklenen dosyayı geçici olarak QUERY_DIR klasörüne kaydet (Ajanlar oradan okuyor)
        os.makedirs(QUERY_DIR, exist_ok=True)
        query_path = os.path.join(QUERY_DIR, uploaded_file.name)
        with open(query_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 2. MAS Süreç Gösterimi
        st.write("### 🔄 Otonom Süreç İzleme")
        with st.status("🤖 Ajanlar görevi devralıyor...", expanded=True) as status_container:
            # Mevcut mimariyi bozmadan Supervisor'ı başlat
            supervisor = SupervisorAgent(image_path=query_path)
            
            # DİNAMİK LOGLAMA: Ajan koduna dokunmadan BlackBoard'un log metodunu monkey-patch ile araya giriyoruz
            original_log = supervisor.bb.log
            def st_log(agent_name: str, message: str):
                original_log(agent_name, message)  # Orijinal işleyişi bozma
                # UI üzerinde canlı göster
                status_container.write(f"**[{agent_name}]** ➾ {message}")
            
            supervisor.bb.log = st_log
            
            # Görevi koştur (Senkron)
            blackboard = supervisor.run_mission()
            
            status_container.update(label="Tüm ajan süreçleri tamamlandı!", state="complete", expanded=False)
        
        # 3. Sonuç Gösterimi
        st.markdown("---")
        st.write("### 📋 Eşleşme Sonucu")
        
        if blackboard.mission_status == "SUCCESS":
            result = blackboard.match_result
            match_name = result.get("name", "Bilinmiyor")
            match_score = result.get("score", 0.0)
            match_status = result.get("status", "Bilinmiyor")
            
            res_col1, res_col2, res_col3 = st.columns([1.5, 1.5, 1])
            
            with res_col1:
                st.image(image, caption="Sorgu Görseli", use_container_width=True)
                
            with res_col2:
                if match_name != "Kayıtlı değil":
                    db_person_dir = os.path.join(DATABASE_DIR, match_name)
                    db_image_path = None
                    # Eşleşen bireyin veritabanındaki ilk fotoğrafını bulup gösterelim
                    if os.path.exists(db_person_dir):
                        db_files = [f for f in os.listdir(db_person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if db_files:
                            db_image_path = os.path.join(db_person_dir, db_files[0])
                    
                    if db_image_path:
                        st.image(db_image_path, caption=f"Eşleşen Veritabanı Kaydı: {match_name}", use_container_width=True)
                    else:
                        st.warning("Veritabanında referans görsel bulunamadı.")
                else:
                    st.info("Sistemde eşleşen birey bulunamadı (Yeni Birey).")
                    
            with res_col3:
                st.metric(label="🐢 Eşleşen ID", value=match_name)
                st.metric(label="📊 Benzerlik Skoru", value=f"%{match_score * 100:.1f}")
                st.metric(label="🏷️ Durum", value=match_status)
                
            # Geri bildirim mesajları
            if match_status == "GÜÇLÜ_EŞLEŞME":
                st.success("✨ Güçlü eşleşme sağlandı! Kimlik doğrulandı.")
            elif match_status == "OLASI_EŞLEŞME":
                st.warning("⚠️ Olası eşleşme tespit edildi. İnsan onayı önerilir.")
            else:
                st.info("🆕 Yeni bir kaplumbağa tespit edildi.")
                
        else:
            st.error(f"❌ Görev Başarısız: {blackboard.error_message}")
            st.info("Detaylar için yukarıdaki 'Otonom Süreç İzleme' paneline tıklayıp hata veren ajanı inceleyebilirsiniz.")
