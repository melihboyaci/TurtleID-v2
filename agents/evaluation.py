"""
evaluation.py — Eşleşme Puanlama Ajanı (EvaluationWorker)
===========================================================

Pipeline'ın beşinci adımı. RecognitionWorker tarafından üretilen
query embedding'i ile veritabanındaki tüm birey embedding'lerini
cosine similarity ile karşılaştırır.

Eşleşme Stratejisi — Max-of-Images:
    Her veritabanı bireyi birden fazla görsel ile temsil edilir.
    Ortalama embedding yerine, query'nin her bir görselle ayrı ayrı
    benzerliği hesaplanır ve **en yüksek** benzerlik skoru o bireyin
    nihai skoru olarak alınır. Bu yaklaşım, averaging-blur problemini
    önler ve probe/galeri paradigmasına uygundur.

BlackBoard Akışı:
    Okur  : query_embedding, db_embeddings, db_files
    Yazar : match_result  →  {"name", "score", "status", "profile_note"}

# ─────────────────────────────────────────────────────────────
# SOLID / Clean Code Uyum Notu
# ─────────────────────────────────────────────────────────────
# SRP  : Yalnızca benzerlik puanlama ve eşleşme kararı verir.
#        Embedding üretimi RecognitionWorker'ın, tensör hazırlığı
#        PreprocessingWorker'ın sorumluluğundadır.
# OCP  : Eşik değerleri config.py'den alınır; farklı bir eşleşme
#        stratejisi eklemek bu sınıfı değiştirmez.
# DIP  : Eşik sabitleri config modülünden enjekte edilir.
# ─────────────────────────────────────────────────────────────
"""

import numpy as np

from agents import BaseWorker
from config import MATCH_THRESHOLD, POSSIBLE_THRESHOLD


class EvaluationWorker(BaseWorker):
    """
    Cosine similarity ile en yakın birey eşleşmesini bulur.

    Skorlama sonucu üç kategoriden birine atanır:
        - GÜÇLÜ_EŞLEŞME : skor ≥ MATCH_THRESHOLD
        - OLASI_EŞLEŞME  : POSSIBLE_THRESHOLD ≤ skor < MATCH_THRESHOLD
        - YENİ_BİREY     : skor < POSSIBLE_THRESHOLD
    """

    def execute(self) -> bool:
        """
        Tüm veritabanı bireyleriyle cosine similarity hesaplar.

        Returns:
            True: Eşleşme sonucu BlackBoard'a yazıldı.
            False: Gerekli embedding verisi eksik.
        """
        if self.bb.query_embedding is None or not self.bb.db_embeddings:
            self.bb.fail(self.name, "Embedding verisi eksik.")
            return False

        q = np.asarray(self.bb.query_embedding)
        q_norm = q / (np.linalg.norm(q) + 1e-12)

        all_scores: list[tuple[str, float]] = []
        for i, db_emb in enumerate(self.bb.db_embeddings):
            db_arr = np.asarray(db_emb)
            # Geriye uyumluluk: tek vektör (1D) ise 2D'ye yükselt
            if db_arr.ndim == 1:
                db_arr = db_arr[np.newaxis, :]
            db_norms = db_arr / (np.linalg.norm(db_arr, axis=1, keepdims=True) + 1e-12)
            sims = db_norms @ q_norm  # (n_images,)
            score = float(sims.max())
            all_scores.append((self.bb.db_files[i], score))

        all_scores.sort(key=lambda x: x[1], reverse=True)

        # DIAGNOSTIC: TOP-5 eşleşme ve dağılım istatistikleri
        self.log("--- TOP-5 EŞLEŞMELER ---")
        for rank, (name, score) in enumerate(all_scores[:5], 1):
            self.log(f"  #{rank}: {name:20s} | %{score*100:.2f}")

        scores_only = [s for _, s in all_scores]
        self.log(
            f"--- DAĞILIM: ort=%{np.mean(scores_only)*100:.2f} "
            f"std=%{np.std(scores_only)*100:.2f} "
            f"min=%{np.min(scores_only)*100:.2f} "
            f"max=%{np.max(scores_only)*100:.2f}"
        )

        best_name, best_score = all_scores[0]

        if best_score >= MATCH_THRESHOLD:
            status = "GÜÇLÜ_EŞLEŞME"
        elif best_score >= POSSIBLE_THRESHOLD:
            status = "OLASI_EŞLEŞME"
        else:
            status = "YENİ_BİREY"
            best_name = "Kayıtlı değil"

        self.bb.match_result = {
            "name": best_name,
            "score": best_score,
            "status": status,
            "profile_note": "Max-of-images yaklaşımı: galeri bazlı en yüksek benzerlik kullanıldı",
        }
        self.log(f"Sonuç: {best_name} | Skor: %{best_score*100:.1f} | {status}")
        return True
