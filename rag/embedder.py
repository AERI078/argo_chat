# rag/embedder.py — converts text to vectors using sentence-transformers

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True).astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
