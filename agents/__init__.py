from abc import ABC, abstractmethod
from blackboard import BlackBoard


class BaseWorker(ABC):
    """
    Tüm worker ajanların miras aldığı soyut sınıf.
    SOLID - DIP: SupervisorAgent somut sınıflara değil
    bu arayüze bağımlıdır.
    """

    def __init__(self, blackboard: BlackBoard):
        self.bb = blackboard  # Shared blackboard referansı

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def execute(self) -> bool:
        """
        Worker'ın görevini yapar. Sonucu blackboard'a yazar.
        Başarılı: True, Başarısız: False döner.
        """
        pass

    def log(self, message: str) -> None:
        self.bb.log(self.name, message)
