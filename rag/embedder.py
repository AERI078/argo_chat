# rag/embedder.py — text to vectors using fastembed
# fastembed has no torch dependency — keeps Docker image under 500MB
# uses the same underlying model (all-MiniLM-L6-v2) so embeddings are compatible
 
import numpy as np
from fastembed import TextEmbedding
from config import EMBED_MODEL
 
 
class Embedder:
    def __init__(self):
        # fastembed downloads model on first use, caches to ~/.cache/fastembed
        self.model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings, dtype="float32")
 
    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
