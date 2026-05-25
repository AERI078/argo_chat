# rag/pipeline.py — orchestrates the full RAG flow
# startup order:
#   1. try download float index from Supabase Storage
#   2. if found → load it (fast, ~5-10s)
#   3. if not found → fetch Argo data → build index → upload to Supabase (slow, once only)

import os
from dataclasses import dataclass, field
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.summarizer import summarize_profiles
from rag.knowledge_pipeline import KnowledgePipeline
from data_pipeline.fetcher import fetch_indian_ocean_profiles
from data_pipeline.db import setup_tables, cache_profiles
from config import FAISS_INDEX_PATH, FAISS_DOCS_PATH, TOP_K

# Supabase Storage remote filenames — must match what was manually uploaded
_REMOTE_FAISS = "floatchat.faiss"
_REMOTE_DOCS  = "floatchat_docs.json"


@dataclass
class RAGResult:
    query: str
    docs: list[str]                                      # float profile summaries
    knowledge_docs: list[str] = field(default_factory=list)  # oceanographic context


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = self._load_or_build_float_index()
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.knowledge = KnowledgePipeline(self.embedder)

    def _load_or_build_float_index(self) -> VectorStore:
        # ── step 1: already on local disk (same container session) ────────────
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_DOCS_PATH):
            print("Loading float index from local disk...")
            return VectorStore.load()

        # ── step 2: try downloading from Supabase Storage ─────────────────────
        print("Float index not on disk — attempting Supabase download...")
        downloaded = VectorStore.download_from_supabase(
            remote_faiss=_REMOTE_FAISS,
            remote_docs=_REMOTE_DOCS,
            local_faiss=FAISS_INDEX_PATH,
            local_docs=FAISS_DOCS_PATH,
        )
        if downloaded:
            print("Float index loaded from Supabase Storage.")
            return VectorStore.load()

        # ── step 3: cold start — build from scratch then upload ───────────────
        print("No index found anywhere — fetching Argo data and building FAISS index...")
        print("This will take ~90 seconds. It only happens once.")
        setup_tables()
        df = fetch_indian_ocean_profiles()
        cache_profiles(df)

        summaries = summarize_profiles(df)
        embeddings = self.embedder.embed(summaries)

        dim = embeddings.shape[1]
        store = VectorStore(dim=dim)
        store.add(embeddings, summaries)
        store.save()
        print(f"Float index built with {len(summaries)} summaries.")

        # upload so future startups skip this step
        print("Uploading float index to Supabase Storage for future startups...")
        VectorStore.upload_to_supabase(
            remote_faiss=_REMOTE_FAISS,
            remote_docs=_REMOTE_DOCS,
            local_faiss=FAISS_INDEX_PATH,
            local_docs=FAISS_DOCS_PATH,
        )
        return store

    def retrieve(self, query: str, k: int = TOP_K) -> RAGResult:
        """
        Retrieves from both indexes.
        Float docs: real measurement summaries for data grounding.
        Knowledge docs: oceanographic context for insight and explanation.
        """
        float_docs = self.retriever.retrieve(query, k=k)
        knowledge_docs = self.knowledge.retrieve(query, k=3)
        return RAGResult(query=query, docs=float_docs, knowledge_docs=knowledge_docs)