# rag/retriever.py — semantic retrieval with dynamic TOP_K and keyword re-ranking
#
# Dynamic TOP_K:
#   simple queries  ("what is a thermocline")        → k=4
#   comparison queries ("why saltier than")           → k=7
#   data/measurement queries ("show profiles near")   → k=6
#   default                                           → k=5
#
# Re-ranking:
#   FAISS returns docs by vector distance (semantic similarity).
#   We then score each doc by keyword overlap with the query and
#   blend the two scores so docs that are both semantically close
#   AND contain the right terms float to the top.

import re
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from config import TOP_K


# query-type signals → how many docs to pull from FAISS before re-ranking
_COMPARISON_WORDS  = {"why", "compare", "difference", "versus", "vs", "between",
                       "contrast", "saltier", "warmer", "cooler", "higher", "lower"}
_DATA_WORDS        = {"show", "plot", "profile", "measurement", "data", "fetch",
                       "lat", "lon", "latitude", "longitude", "float", "temperature",
                       "salinity", "depth", "pressure"}
_CONCEPTUAL_WORDS  = {"what", "how", "explain", "define", "mean", "cause",
                       "effect", "impact", "role", "why"}


def _dynamic_k(query: str) -> int:
    """
    Returns the number of docs to retrieve based on detected query complexity.
    Over-fetching is cheap; under-fetching loses context.
    """
    tokens = set(query.lower().split())

    if tokens & _COMPARISON_WORDS:
        return 8   # comparisons need context from multiple regions
    if tokens & _DATA_WORDS:
        return 6   # data queries need a few concrete float summaries
    if tokens & _CONCEPTUAL_WORDS:
        return 5   # conceptual questions need focused knowledge chunks
    return TOP_K   # fallback to config default


def _keyword_score(doc: str, query_terms: set[str]) -> float:
    """
    Fraction of meaningful query terms that appear in the doc.
    Ignores stopwords — only content words contribute to the score.
    Returns 0.0–1.0.
    """
    _STOPWORDS = {"what", "is", "a", "the", "of", "in", "to", "and", "or",
                   "for", "are", "how", "does", "why", "it", "be", "was",
                   "that", "this", "with", "from", "by", "an", "at", "on"}
    content_terms = query_terms - _STOPWORDS
    if not content_terms:
        return 0.0
    doc_lower = doc.lower()
    hits = sum(1 for term in content_terms if term in doc_lower)
    return hits / len(content_terms)


def _rerank(docs: list[str], query: str, top_n: int) -> list[str]:
    """
    Blends FAISS rank (position-based) with keyword overlap score.
    FAISS already sorted by semantic similarity so earlier = better.
    We use a 60/40 blend: 60% semantic rank, 40% keyword overlap.

    Returns top_n docs sorted by blended score descending.
    """
    if not docs:
        return docs

    query_terms = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    n = len(docs)

    scored = []
    for rank, doc in enumerate(docs):
        # semantic score: 1.0 for rank 0 (best FAISS match), decreasing linearly
        semantic_score = 1.0 - (rank / n)
        keyword_score  = _keyword_score(doc, query_terms)
        blended        = (0.6 * semantic_score) + (0.4 * keyword_score)
        scored.append((blended, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder     = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = TOP_K) -> list[str]:
        """
        1. Determine fetch_k: either the caller-supplied k or dynamic k,
           whichever is larger — we always over-fetch then trim via re-ranking.
        2. Embed query and search FAISS for fetch_k docs.
        3. Re-rank by blended semantic + keyword score.
        4. Return top k docs.
        """
        fetch_k  = max(k, _dynamic_k(query))
        query_vec = self.embedder.embed_one(query)
        raw_docs  = self.vector_store.search(query_vec, k=fetch_k)
        return _rerank(raw_docs, query, top_n=k)