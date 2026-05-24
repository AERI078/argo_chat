# rag/retriever.py — given a query string, returns relevant float summaries

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from config import TOP_K


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = TOP_K) -> list[str]:
        query_vec = self.embedder.embed_one(query)
        return self.vector_store.search(query_vec, k=k)
