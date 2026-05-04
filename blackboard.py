"""
Tüm ajanların okuyup yazabildiği merkezi durum deposu.
Hierarchical multi-agent mimarisinin kalbi.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BlackBoard:
    # --- GÖREV GİRDİSİ ---
    query_image_path: str = ""

    # --- AJAN ÇIKTILARI (Sırayla dolar) ---
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
        """Tüm ajanlar bu metodla blackboard'a log yazar."""
        from datetime import datetime
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_name}] {message}"
        self.mission_log.append(entry)
        print(entry)

    def set_step(self, step: str) -> None:
        self.current_step = step

    def fail(self, agent_name: str, reason: str) -> None:
        self.mission_status = "FAILED"
        self.error_message = reason
        self.log(agent_name, f"MİSYON BAŞARISIZ: {reason}")
