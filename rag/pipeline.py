# rag/pipeline.py — orchestrates the full RAG flow and builds the index on first run

import os
from dataclasses import dataclass
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.summarizer import summarize_profiles
from data_pipeline.fetcher import fetch_indian_ocean_profiles
from data_pipeline.db import setup_tables, cache_profiles
from config import FAISS_INDEX_PATH, FAISS_DOCS_PATH, TOP_K


@dataclass
class RAGResult:
    query: str
    docs: list[str]      # retrieved float summaries — passed to agent as context


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = self._load_or_build_index()
        self.retriever = Retriever(self.embedder, self.vector_store)

    def _load_or_build_index(self) -> VectorStore:
        # if index already exists on disk, load it — avoids re-fetching on every restart
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_DOCS_PATH):
            print("Loading existing FAISS index...")
            return VectorStore.load()

        print("No index found — fetching Argo data and building FAISS index...")
        setup_tables()
        df = fetch_indian_ocean_profiles()
        cache_profiles(df)

        summaries = summarize_profiles(df)
        embeddings = self.embedder.embed(summaries)

        dim = embeddings.shape[1]
        store = VectorStore(dim=dim)
        store.add(embeddings, summaries)
        store.save()
        print(f"Index built with {len(summaries)} float profile summaries.")
        return store

    def retrieve(self, query: str, k: int = TOP_K) -> RAGResult:
        docs = self.retriever.retrieve(query, k=k)
        return RAGResult(query=query, docs=docs)
