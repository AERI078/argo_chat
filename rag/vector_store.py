# rag/vector_store.py — builds, saves, loads, and searches the FAISS index

import json
import numpy as np
import faiss
from config import FAISS_INDEX_PATH, FAISS_DOCS_PATH, TOP_K


class VectorStore:
    def __init__(self, dim: int):
        # FlatL2 — exact search, no approximation
        # right choice for our dataset size (thousands of profiles, not millions)
        self.index = faiss.IndexFlatL2(dim)
        self.docs: list[str] = []

    def add(self, embeddings: np.ndarray, docs: list[str]):
        self.index.add(embeddings)
        self.docs.extend(docs)

    def search(self, query_embedding: np.ndarray, k: int = TOP_K) -> list[str]:
        _, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.docs[i] for i in indices[0] if i < len(self.docs)]

    def save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(FAISS_DOCS_PATH, "w") as f:
            json.dump(self.docs, f)

    @classmethod
    def load(cls) -> "VectorStore":
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_DOCS_PATH) as f:
            docs = json.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.docs = docs
        return store

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0
