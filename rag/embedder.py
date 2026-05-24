# rag/embedder.py — text to vectors using fastembed
# no torch dependency — keeps Docker image under 500MB
# same model as before (all-MiniLM-L6-v2) so FAISS index stays compatible

import numpy as np
from fastembed import TextEmbedding
from config import EMBED_MODEL

# fastembed requires the full HuggingFace model name
_FASTEMBED_MODEL = f"sentence-transformers/{EMBED_MODEL}"


class Embedder:
    def __init__(self):
        self.model = TextEmbedding(model_name=_FASTEMBED_MODEL)

    def embed(self, texts: list[str]) -> np.ndarray:
        # fastembed.embed() returns a generator — convert to array immediately
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings, dtype="float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]