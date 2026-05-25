# rag/vector_store.py — builds, saves, loads, and searches the FAISS index
# also handles upload/download from Supabase Storage so the index
# survives container restarts on Render

import json
import os
import numpy as np
import faiss
from config import FAISS_INDEX_PATH, FAISS_DOCS_PATH, TOP_K

# lazy import — only needed when talking to Supabase Storage
def _get_supabase_client():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)

BUCKET = "floatchat"


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
        """Save index to local disk."""
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(FAISS_DOCS_PATH, "w") as f:
            json.dump(self.docs, f)

    @classmethod
    def load(cls) -> "VectorStore":
        """Load index from local disk."""
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

    # ── Supabase Storage helpers ───────────────────────────────────────────────

    @staticmethod
    def download_from_supabase(remote_faiss: str, remote_docs: str,
                                local_faiss: str, local_docs: str) -> bool:
        """
        Downloads index files from Supabase Storage bucket to local disk.
        Returns True if both files downloaded successfully, False otherwise.
        """
        try:
            sb = _get_supabase_client()

            print(f"Downloading {remote_faiss} from Supabase...")
            faiss_bytes = sb.storage.from_(BUCKET).download(remote_faiss)
            with open(local_faiss, "wb") as f:
                f.write(faiss_bytes)

            print(f"Downloading {remote_docs} from Supabase...")
            docs_bytes = sb.storage.from_(BUCKET).download(remote_docs)
            with open(local_docs, "wb") as f:
                f.write(docs_bytes)

            print(f"Downloaded {local_faiss} ({len(faiss_bytes)/1024:.1f} KB) "
                  f"and {local_docs} ({len(docs_bytes)/1024:.1f} KB)")
            return True

        except Exception as e:
            print(f"Supabase download failed: {e}")
            # clean up partial downloads so we don't load a corrupt index
            for path in [local_faiss, local_docs]:
                if os.path.exists(path):
                    os.remove(path)
            return False

    @staticmethod
    def upload_to_supabase(remote_faiss: str, remote_docs: str,
                            local_faiss: str, local_docs: str):
        """
        Uploads local index files to Supabase Storage bucket.
        Called once after a fresh index build so future startups can skip the build.
        Uses upsert so re-uploading an updated index doesn't fail.
        """
        try:
            sb = _get_supabase_client()

            for local_path, remote_path in [(local_faiss, remote_faiss),
                                             (local_docs, remote_docs)]:
                with open(local_path, "rb") as f:
                    data = f.read()
                # upsert=True overwrites if file already exists in bucket
                sb.storage.from_(BUCKET).upload(
                    path=remote_path,
                    file=data,
                    file_options={"upsert": "true"}
                )
                print(f"Uploaded {local_path} → supabase://{BUCKET}/{remote_path} "
                      f"({len(data)/1024:.1f} KB)")

        except Exception as e:
            # upload failure is non-fatal — index is still usable locally
            print(f"Supabase upload failed (non-fatal): {e}")