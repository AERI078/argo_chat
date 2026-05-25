# rag/knowledge_pipeline.py — builds and searches a FAISS index over local knowledge docs
# startup order mirrors pipeline.py:
#   1. try local disk
#   2. try Supabase Storage download
#   3. build from knowledge/ folder and upload

import os
import json
import numpy as np
from pathlib import Path
from rag.embedder import Embedder
from rag.vector_store import VectorStore

KNOWLEDGE_DIR         = "knowledge"
KNOWLEDGE_INDEX_PATH  = "knowledge.faiss"
KNOWLEDGE_DOCS_PATH   = "knowledge_docs.json"
CHUNK_SIZE            = 400   # chars — focused but with enough context
CHUNK_OVERLAP         = 80

# Supabase Storage remote filenames — must match what was manually uploaded
_REMOTE_FAISS = "knowledge.faiss"
_REMOTE_DOCS  = "knowledge_docs.json"


def _chunk_text(text: str) -> list[str]:
    """
    Splits a document into overlapping chunks.
    Overlap ensures context isn't lost at boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]  # drop tiny trailing chunks


def _load_knowledge_docs() -> list[str]:
    """Reads all .txt files from the knowledge/ folder and chunks them."""
    chunks = []
    knowledge_path = Path(KNOWLEDGE_DIR)
    if not knowledge_path.exists():
        print(f"Warning: knowledge/ folder not found at {KNOWLEDGE_DIR}")
        return []
    for filepath in sorted(knowledge_path.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        file_chunks = _chunk_text(text)
        source = filepath.stem.replace("_", " ").title()
        chunks.extend([f"[{source}] {chunk}" for chunk in file_chunks])
    print(f"Knowledge base: loaded {len(chunks)} chunks from {knowledge_path}")
    return chunks


class KnowledgePipeline:
    """
    Separate RAG pipeline for oceanographic knowledge documents.
    Used alongside the float summaries index to answer conceptual questions.
    """
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.vector_store = self._load_or_build_index()

    def _load_or_build_index(self) -> VectorStore:
        # ── step 1: already on local disk ─────────────────────────────────────
        if os.path.exists(KNOWLEDGE_INDEX_PATH) and os.path.exists(KNOWLEDGE_DOCS_PATH):
            print("Loading knowledge index from local disk...")
            index = __import__("faiss").read_index(KNOWLEDGE_INDEX_PATH)
            with open(KNOWLEDGE_DOCS_PATH) as f:
                docs = json.load(f)
            store = VectorStore(dim=index.d)
            store.index = index
            store.docs = docs
            return store

        # ── step 2: try downloading from Supabase Storage ─────────────────────
        print("Knowledge index not on disk — attempting Supabase download...")
        downloaded = VectorStore.download_from_supabase(
            remote_faiss=_REMOTE_FAISS,
            remote_docs=_REMOTE_DOCS,
            local_faiss=KNOWLEDGE_INDEX_PATH,
            local_docs=KNOWLEDGE_DOCS_PATH,
        )
        if downloaded:
            print("Knowledge index loaded from Supabase Storage.")
            index = __import__("faiss").read_index(KNOWLEDGE_INDEX_PATH)
            with open(KNOWLEDGE_DOCS_PATH) as f:
                docs = json.load(f)
            store = VectorStore(dim=index.d)
            store.index = index
            store.docs = docs
            return store

        # ── step 3: build from knowledge/ folder and upload ───────────────────
        print("Building knowledge index from local documents...")
        chunks = _load_knowledge_docs()

        if not chunks:
            # return empty store if no docs — system still works without it
            dummy_vec = self.embedder.embed_one("placeholder")
            return VectorStore(dim=len(dummy_vec))

        embeddings = self.embedder.embed(chunks)
        store = VectorStore(dim=embeddings.shape[1])
        store.add(embeddings, chunks)

        # save locally
        import faiss as _faiss
        _faiss.write_index(store.index, KNOWLEDGE_INDEX_PATH)
        with open(KNOWLEDGE_DOCS_PATH, "w") as f:
            json.dump(store.docs, f)
        print(f"Knowledge index built with {len(chunks)} chunks.")

        # upload so future startups skip this step
        print("Uploading knowledge index to Supabase Storage...")
        VectorStore.upload_to_supabase(
            remote_faiss=_REMOTE_FAISS,
            remote_docs=_REMOTE_DOCS,
            local_faiss=KNOWLEDGE_INDEX_PATH,
            local_docs=KNOWLEDGE_DOCS_PATH,
        )
        return store

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """
        Returns top-k relevant knowledge chunks for a query.
        Over-fetches by 2x then re-ranks by blended semantic + keyword score
        so the best chunks surface regardless of pure vector distance.
        """
        if self.vector_store.is_empty:
            return []
        from rag.retriever import _rerank
        query_vec = self.embedder.embed_one(query)
        raw_docs = self.vector_store.search(query_vec, k=min(k * 2, 10))
        return _rerank(raw_docs, query, top_n=k)