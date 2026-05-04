"""
blackboard.py — Merkezi Durum Deposu (Shared State)
=====================================================

Hiyerarşik Çoklu Ajan Sistemi'nin (Hierarchical MAS) kalbidir.
Tüm ajanlar bu nesne üzerinden okuma/yazma yaparak haberleşir;
birbirine doğrudan referans vermez.

Blackboard Tasarım Deseni (Corkill, 1991):
    Bağımsız uzman ajanlar (Knowledge Sources), paylaşılan bir veri
    deposu (Blackboard) üzerinden asenkron olarak çalışır. Bir ajanın
    çıktısı diğerinin girdisi olur; ancak ajanlar birbirini tanımak
    zorunda değildir.

Veri Akışı:
    query_image_path
        → audit_result
        → head_crop, head_confidence
        → model_ready_tensor
        → query_embedding
        → db_embeddings, db_files
        → match_result

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : BlackBoard yalnızca durum taşır; iş mantığı içermez.
# OCP  : Yeni bir ajan çıktısı eklemek için yalnızca yeni bir alan
#        (field) eklenir; mevcut alanlar değişmez.
# DIP  : Ajanlar somut birbirine değil, bu paylaşılan veri yapısına
#        bağımlıdır → gevşek bağlılık (loose coupling).
# Mediator Pattern: BlackBoard, ajanlar arasında dolaylı iletişim
#        aracı olarak görev yapar.
# ─────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class BlackBoard:
    """
    Tüm ajanların okuyup yazabildiği merkezi durum nesnesi.

    Her ajan, pipeline'daki sıraya göre ilgili alana yazar;
    sonraki ajan o alanı okur. SupervisorAgent yalnızca
    ``mission_status`` ve ``error_message`` alanlarını kontrol
    ederek akışı yönetir.

    Attributes:
        query_image_path: Sorgulanacak görselin dosya yolu.
        audit_result: AuditWorker doğrulama sonucu.
        head_crop: HeadDetectionWorker'ın RGB kafa kırpması.
        head_confidence: Kafa tespitinin güven skoru [0, 1].
        model_ready_tensor: PreprocessingWorker'ın model girdisi.
        query_embedding: RecognitionWorker'ın query vektörü.
        db_embeddings: Veritabanı embedding matrisleri listesi.
        db_files: Veritabanı birey isimleri listesi.
        match_result: EvaluationWorker'ın eşleşme sonucu.
        current_step: Pipeline'daki aktif adım adı.
        mission_status: Görev durumu (PENDING/RUNNING/SUCCESS/FAILED).
        error_message: Hata mesajı (varsa).
        mission_log: Kronolojik log kayıtları.
    """

    # --- GÖREV GİRDİSİ ---
    query_image_path: str = ""

    # --- AJAN ÇIKTILARI (Pipeline sırasına göre dolar) ---
    audit_result: dict = field(default_factory=dict)
    head_crop: Optional[np.ndarray] = None
    head_confidence: float = 0.0
    model_ready_tensor: Optional[np.ndarray] = None
    query_embedding: Optional[np.ndarray] = None
    db_embeddings: list = field(default_factory=list)
    db_files: list = field(default_factory=list)
    match_result: dict = field(default_factory=dict)

    # --- MİSYON DURUMU ---
    current_step: str = "IDLE"
    mission_status: str = "PENDING"  # PENDING, RUNNING, SUCCESS, FAILED
    error_message: str = ""
    mission_log: list = field(default_factory=list)

    def log(self, agent_name: str, message: str) -> None:
        """
        Merkezi loglama metodu.

        Tüm ajanlar bu metotla BlackBoard'a kronolojik log yazar.
        Log aynı zamanda standart çıktıya yazdırılır.

        Args:
            agent_name: Logu yazan ajanın adı.
            message: Log mesajı.
        """
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_name}] {message}"
        self.mission_log.append(entry)
        print(entry)

    def set_step(self, step: str) -> None:
        """Pipeline'daki aktif adımı günceller."""
        self.current_step = step

    def fail(self, agent_name: str, reason: str) -> None:
        """
        Görevi başarısız olarak işaretler.

        Args:
            agent_name: Hatayı raporlayan ajanın adı.
            reason: İnsan tarafından okunabilir hata açıklaması.
        """
        self.mission_status = "FAILED"
        self.error_message = reason
        self.log(agent_name, f"MİSYON BAŞARISIZ: {reason}")
