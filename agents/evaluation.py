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

        # Tüm DB ile skor hesapla, sırala
        # Her DB kaydı artık embedding listesi (n_images x 256).
        # Max-of-images: query'nin o klasördeki herhangi bir görüntüyle
        # en yüksek benzerliği o kaplumbağanın skoru olur.
        import numpy as _np
        q = _np.asarray(self.bb.query_embedding)
        q_norm = q / (_np.linalg.norm(q) + 1e-12)

        all_scores = []
        for i, db_emb in enumerate(self.bb.db_embeddings):
            db_arr = _np.asarray(db_emb)
            # Geriye uyumluluk: tek vektör (1D) ise 2D'ye yükselt
            if db_arr.ndim == 1:
                db_arr = db_arr[_np.newaxis, :]
            db_norms = db_arr / (_np.linalg.norm(db_arr, axis=1, keepdims=True) + 1e-12)
            sims = db_norms @ q_norm  # (n_images,)
            score = float(sims.max())
            all_scores.append((self.bb.db_files[i], score))

        all_scores.sort(key=lambda x: x[1], reverse=True)

        # DIAGNOSTIC: TOP-5 + dağılım istatistikleri
        self.log("--- TOP-5 EŞLEŞMELER ---")
        for rank, (name, score) in enumerate(all_scores[:5], 1):
            self.log(f"  #{rank}: {name:20s} | %{score*100:.2f}")

        scores_only = [s for _, s in all_scores]
        import numpy as _np
        self.log(
            f"--- DAĞILIM: ort=%{_np.mean(scores_only)*100:.2f} "
            f"std=%{_np.std(scores_only)*100:.2f} "
            f"min=%{_np.min(scores_only)*100:.2f} "
            f"max=%{_np.max(scores_only)*100:.2f}"
        )

        best_name, best_score = all_scores[0]

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
