"""
FAISS Vector Store — IndexHNSWFlat (M=32) alternative to ChromaDB.

Switch at runtime with the ``VECTOR_STORE`` environment variable:

    VECTOR_STORE=chromadb   (default) — existing ChromaDB backend
    VECTOR_STORE=faiss                — FAISS HNSW backend

No data migration is required; both paths initialise independently.

Example
-------
    from retrieval.faiss_store import get_vector_store

    store = get_vector_store()          # respects VECTOR_STORE env var
    store.add_documents(texts, metas)   # works identically for both backends
"""

import json
import os
from typing import Dict, List, Optional

import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Default persistence paths
FAISS_INDEX_PATH = os.path.join("models", "faiss_index.bin")
FAISS_META_PATH = os.path.join("models", "faiss_metadata.json")


class FAISSStore:
    """Lightweight FAISS wrapper using IndexHNSWFlat (M=32).

    Parameters
    ----------
    embedding_model_name : str
        HuggingFace model used for embedding (default ``all-MiniLM-L6-v2``).
    m : int
        HNSW graph connectivity parameter (default 32).
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        m: int = 32,
    ):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.m = m
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []
        self._dim: Optional[int] = None

    # ── Build ───────────────────────────────────────────────────────────

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """Embed *texts* and add them to the HNSW index."""
        embeddings = self.embedding_model.embed_documents(texts)
        vectors = np.array(embeddings, dtype="float32")

        if self.index is None:
            self._dim = vectors.shape[1]
            self.index = faiss.IndexHNSWFlat(self._dim, self.m)

        self.index.add(vectors)

        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            self.documents.append({"text": text, "metadata": meta})

    # ── Query ───────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Return the *k* most similar documents to *query*."""
        if self.index is None or self.index.ntotal == 0:
            return []

        q_vec = np.array(
            [self.embedding_model.embed_query(query)], dtype="float32"
        )
        distances, indices = self.index.search(q_vec, k)

        results: List[Dict] = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(dist)
                results.append(doc)
        return results

    # ── Persistence ─────────────────────────────────────────────────────

    def save(
        self,
        index_path: str = FAISS_INDEX_PATH,
        meta_path: str = FAISS_META_PATH,
    ) -> None:
        """Write the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w") as f:
            json.dump(self.documents, f)

    def load(
        self,
        index_path: str = FAISS_INDEX_PATH,
        meta_path: str = FAISS_META_PATH,
    ) -> None:
        """Load a previously persisted index from disk."""
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            self.documents = json.load(f)
        self._dim = self.index.d


# ── Factory ─────────────────────────────────────────────────────────────────

def get_vector_store(embedding_model_name: str = "all-MiniLM-L6-v2"):
    """Return the active vector store based on the ``VECTOR_STORE`` env var.

    Returns
    -------
    FAISSStore | Chroma
        Ready-to-use store instance.  For ``"faiss"`` a fresh (empty)
        ``FAISSStore`` is returned; call ``add_documents`` or ``load``
        to populate it.  For ``"chromadb"`` the existing persisted Chroma
        collection is opened.
    """
    backend = os.environ.get("VECTOR_STORE", "chromadb").strip().lower()

    if backend == "faiss":
        store = FAISSStore(embedding_model_name=embedding_model_name)
        # Auto-load if a persisted index exists
        if os.path.exists(FAISS_INDEX_PATH):
            store.load()
        return store

    # Default: ChromaDB (existing path)
    from langchain_community.vectorstores import Chroma

    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return Chroma(
        persist_directory="models/chroma_db",
        embedding_function=embedding,
    )
