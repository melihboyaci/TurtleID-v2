from scipy.spatial.distance import cosine
from agents import BaseWorker


class EvaluationWorker(BaseWorker):
    """
    Cosine Similarity ile en yakın eşleşmeyi bulur.
    Blackboard'dan embedding'leri okur, match_result yazar.
    """

    MATCH_THRESHOLD: float = 0.85
    POSSIBLE_THRESHOLD: float = 0.70

    def execute(self) -> bool:
        if self.bb.query_embedding is None or not self.bb.db_embeddings:
            self.bb.fail(self.name, "Embedding verisi eksik.")
            return False

        best_score = 0.0
        best_name = "Bilinmiyor"

        for i, db_emb in enumerate(self.bb.db_embeddings):
            score = 1 - cosine(self.bb.query_embedding, db_emb)
            if score > best_score:
                best_score = score
                best_name = self.bb.db_files[i]

        if best_score >= self.MATCH_THRESHOLD:
            status = "GÜÇLÜ_EŞLEŞME"
        elif best_score >= self.POSSIBLE_THRESHOLD:
            status = "OLASI_EŞLEŞME"
        else:
            status = "YENİ_BİREY"
            best_name = "Kayıtlı değil"

        self.bb.match_result = {
            "name": best_name,
            "score": best_score,
            "status": status,
            "profile_note": "Sağ+sol profil ortalaması kullanıldı",
        }
        self.log(f"Sonuç: {best_name} | Skor: %{best_score*100:.1f} | {status}")
        return True
