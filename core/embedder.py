"""
core/embedder.py
================
Thin wrapper around SentenceTransformer for producing dense embeddings.

Singleton pattern — model is loaded once and reused.
All public functions accept plain strings or lists of strings.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL


# ── Singleton ─────────────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("[embedder] Model loaded.")
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def embed(texts: str | list[str], batch_size: int = 256) -> np.ndarray:
    """
    Encode one or more texts into L2-normalised embedding vectors.

    Args:
        texts      : single string or list of strings
        batch_size : how many to encode per forward pass (tune for RAM)

    Returns:
        np.ndarray of shape (N, 384) — float32, L2-normalised
    """
    model = _get_model()
    if isinstance(texts, str):
        texts = [texts]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.
    If vectors are L2-normalised this is just the dot product.
    """
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between a single query vector and each row in matrix.

    Args:
        query  : shape (D,)
        matrix : shape (N, D)

    Returns:
        shape (N,) — float32
    """
    query = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms
    return (normed @ query).astype(np.float32)


if __name__ == "__main__":
    v1 = embed("I love hiking in the mountains")
    v2 = embed("Trekking through forests is my passion")
    v3 = embed("I enjoy cooking pasta")
    print(f"Similar pair sim   : {cosine_similarity(v1[0], v2[0]):.4f}")   # expect ~0.7+
    print(f"Different pair sim : {cosine_similarity(v1[0], v3[0]):.4f}")   # expect <0.4
