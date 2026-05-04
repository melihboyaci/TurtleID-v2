"""
agents — TurtleID Çoklu Ajan Sistemi (Multi-Agent System) Paketi
================================================================

Bu paket, Hiyerarşik MAS mimarisinin worker ajanlarını içerir.
Tüm worker'lar ``BaseWorker`` soyut sınıfından türer ve
``BlackBoard`` üzerinden haberleşir.

Paket Yapısı:
    BaseWorker          — Tüm worker'ların soyut ata sınıfı (bu dosya)
    AuditWorker         — Girdi dosyası doğrulama
    HeadDetectionWorker — Gemini Vision ile kafa profili doğrulama
    PreprocessingWorker — Tensör hazırlığı (model girdisi)
    RecognitionWorker   — Embedding çıkarma (Triplet-Loss modeli)
    EvaluationWorker    — Cosine similarity ile eşleşme puanlama
    ReportingWorker     — Gemini LLM ile rapor üretimi

Mimari Kurallar:
    1. Worker'lar birbirini doğrudan çağırmaz (loose coupling).
    2. Veri akışı yalnızca BlackBoard üzerinden gerçekleşir.
    3. SupervisorAgent, worker'ları sırayla delege eder.

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : BaseWorker yalnızca soyut kontrat ve ortak log altyapısını sağlar.
# OCP  : Yeni bir worker eklemek için BaseWorker'dan türemek yeterlidir;
#        mevcut worker'lar değişmez.
# LSP  : Her alt sınıf, execute() → bool kontratına uyar.
# ISP  : BaseWorker minimum arayüz sunar (execute + log).
# DIP  : SupervisorAgent somut worker sınıflarına değil BaseWorker
#        arayüzüne bağımlıdır.
# ─────────────────────────────────────────────────────────────
"""

from abc import ABC, abstractmethod
from blackboard import BlackBoard


class BaseWorker(ABC):
    """
    Tüm worker ajanların miras aldığı soyut temel sınıf.

    SupervisorAgent, somut worker sınıflarını değil bu soyut arayüzü
    kullanarak görev delege eder (Dependency Inversion Principle).

    Attributes:
        bb: Paylaşılan BlackBoard referansı. Worker tüm girdi/çıktı
            verilerini bu nesne üzerinden okur ve yazar.
    """

    def __init__(self, blackboard: BlackBoard) -> None:
        self.bb = blackboard

    @property
    def name(self) -> str:
        """Worker'ın sınıf adını döndürür (loglama için)."""
        return self.__class__.__name__

    @abstractmethod
    def execute(self) -> bool:
        """
        Worker'ın asıl görevini yerine getirir.

        Alt sınıflar bu metodu implemente ederek kendi görevlerini tanımlar.
        Sonuçlar doğrudan BlackBoard'a yazılır.

        Returns:
            True: Görev başarıyla tamamlandı.
            False: Görev başarısız oldu (hata BlackBoard'a yazılmıştır).
        """
        ...

    def log(self, message: str) -> None:
        """BlackBoard'un merkezi log mekanizmasına mesaj yazar."""
        self.bb.log(self.name, message)
