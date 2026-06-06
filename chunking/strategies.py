"""
Chunking Strategies for the Financial Document Intelligence RAG Pipeline.

Two approaches benchmarked against the existing evaluation set:
  1. fixed_size_chunk  — deterministic 512-token windows with 50-token overlap
  2. semantic_chunk    — sentence-transformers cosine-similarity splitting (θ = 0.85)

Run as a script to reproduce the benchmark:
    python -m chunking.strategies
"""

import json
import os
import re
import sqlite3
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


# ── helpers ─────────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _split_sentences(text: str) -> List[str]:
    """Lightweight regex sentence splitter."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in parts if s]


# ── 1. Fixed-Size Chunking (512 tokens, 50-token overlap) ──────────────────

def fixed_size_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """Split *text* into whitespace-token windows of *chunk_size* with *overlap*.

    Parameters
    ----------
    text : str
        Raw document text.
    chunk_size : int
        Number of whitespace tokens per chunk (default 512).
    overlap : int
        Overlapping tokens between consecutive chunks (default 50).

    Returns
    -------
    list[str]
        Ordered list of chunk strings.
    """
    tokens = text.split()
    if len(tokens) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks


# ── 2. Semantic Chunking (sentence-transformers, cosine threshold 0.85) ────

def semantic_chunk(
    text: str,
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[str]:
    """Split *text* at semantic boundaries detected via cosine similarity.

    Consecutive sentences whose embedding similarity drops below *threshold*
    trigger a new chunk.  Uses ``sentence-transformers`` for encoding.

    Parameters
    ----------
    text : str
        Raw document text.
    threshold : float
        Cosine similarity cutoff (default 0.85).
    model_name : str
        HuggingFace sentence-transformer model name.

    Returns
    -------
    list[str]
        Semantically coherent chunk strings.
    """
    from sentence_transformers import SentenceTransformer

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks: List[str] = []
    current: List[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))
    return chunks


# ── 3. Precision@5 Benchmark ───────────────────────────────────────────────

def _load_eval_corpus() -> Tuple[List[str], List[dict]]:
    """Load historical fraud cases from the existing SQLite eval set."""
    db_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "fraud_cases.db")
    )
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Eval database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM historical_fraud LIMIT 200", conn)
    conn.close()

    texts, metas = [], []
    for _, row in df.iterrows():
        content = (
            f"User ID: {row['user_id']} | "
            f"Amount: ${row['purchase_value']} | "
            f"Device: {row['device_id']} | "
            f"IP: {row['ip_address']} | "
            f"Browser: {row['browser']} | "
            f"Source: {row['source']} | "
            f"Resolution: {row['resolution_notes']}"
        )
        texts.append(content)
        metas.append({
            "user_id": str(row["user_id"]),
            "purchase_value": float(row["purchase_value"]),
            "is_fraud": int(row["class"]),
        })
    return texts, metas


def _precision_at_k(relevant: set, retrieved: List[str], k: int = 5) -> float:
    """Compute Precision@k given a set of relevant doc IDs."""
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def run_benchmark() -> Dict:
    """Run Precision@5 for both chunking strategies and persist results.

    The benchmark:
      1. Loads 200 historical fraud cases from SQLite.
      2. Chunks them with *fixed_size_chunk* and *semantic_chunk*.
      3. Embeds every chunk with all-MiniLM-L6-v2.
      4. For a set of synthetic probe queries, retrieves top-5 chunks
         by cosine similarity and computes Precision@5.
      5. Saves results to ``chunking/benchmark_results.json``.
    """
    from sentence_transformers import SentenceTransformer

    print("Loading eval corpus …")
    texts, metas = _load_eval_corpus()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # -- Chunking --
    fixed_chunks, fixed_ids = [], []
    semantic_chunks_list, semantic_ids = [], []

    for idx, text in enumerate(texts):
        fc = fixed_size_chunk(text)
        for c in fc:
            fixed_chunks.append(c)
            fixed_ids.append(str(metas[idx]["user_id"]))

        sc = semantic_chunk(text, threshold=0.85, model_name="all-MiniLM-L6-v2")
        for c in sc:
            semantic_chunks_list.append(c)
            semantic_ids.append(str(metas[idx]["user_id"]))

    print(f"Fixed chunks: {len(fixed_chunks)}  |  Semantic chunks: {len(semantic_chunks_list)}")

    # -- Embed --
    print("Embedding fixed-size chunks …")
    fixed_vecs = model.encode(fixed_chunks, show_progress_bar=True)
    print("Embedding semantic chunks …")
    semantic_vecs = model.encode(semantic_chunks_list, show_progress_bar=True)

    # -- Probe queries (derived from known fraud patterns in the dataset) --
    queries = [
        "rapid IP address changes indicating fraud ring activity",
        "high purchase value shortly after signup suspicious transaction",
        "device velocity anomaly multiple devices same user",
        "confirmed fraud based on suspicious IPs or device velocity",
        "unusual browser switching pattern linked to fraud",
    ]

    # Relevant = any chunk from a confirmed fraud case (is_fraud == 1)
    fraud_user_ids = {m["user_id"] for m in metas if m["is_fraud"] == 1}

    def _retrieve_top_k(q_vec, chunk_vecs, chunk_doc_ids, k=5):
        sims = np.dot(chunk_vecs, q_vec) / (
            np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-10
        )
        top_indices = np.argsort(sims)[::-1][:k]
        return [chunk_doc_ids[i] for i in top_indices]

    # -- Evaluate --
    fixed_p5_scores, semantic_p5_scores = [], []

    for q in queries:
        q_vec = model.encode([q])[0]

        ret_fixed = _retrieve_top_k(q_vec, fixed_vecs, fixed_ids, k=5)
        ret_semantic = _retrieve_top_k(q_vec, semantic_vecs, semantic_ids, k=5)

        fixed_p5_scores.append(_precision_at_k(fraud_user_ids, ret_fixed, k=5))
        semantic_p5_scores.append(_precision_at_k(fraud_user_ids, ret_semantic, k=5))

    results = {
        "fixed_size_chunk": {
            "config": {"chunk_size": 512, "overlap": 50},
            "total_chunks": len(fixed_chunks),
            "precision_at_5_per_query": fixed_p5_scores,
            "precision_at_5_mean": round(float(np.mean(fixed_p5_scores)), 4),
        },
        "semantic_chunk": {
            "config": {"model": "all-MiniLM-L6-v2", "cosine_threshold": 0.85},
            "total_chunks": len(semantic_chunks_list),
            "precision_at_5_per_query": semantic_p5_scores,
            "precision_at_5_mean": round(float(np.mean(semantic_p5_scores)), 4),
        },
        "num_eval_documents": len(texts),
        "num_queries": len(queries),
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Benchmark results saved to {out_path}")
    print(json.dumps(results, indent=4))
    return results


# ── CLI entry-point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_benchmark()
