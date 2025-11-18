# reranker_ce.py
#
# Thin wrapper around a sentence-transformers CrossEncoder for reranking.

from pathlib import Path
from typing import List
from sentence_transformers import CrossEncoder


class RerankerCE:
    def __init__(self, ckpt_dir: str):
        ckpt = Path(ckpt_dir)
        assert ckpt.exists(), f"Checkpoint not found at {ckpt}"
        # num_labels=1 for regression scoring, max_length tune if needed
        self.model = CrossEncoder(str(ckpt), num_labels=1, max_length=256)
        print("Reranker loaded:", ckpt)

    def score(self, query: str, docs: List[str]) -> List[float]:
        """
        Score a list of documents for a single query.
        Returns a list of floats (higher = more relevant).
        """
        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

