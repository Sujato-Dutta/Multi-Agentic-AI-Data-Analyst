"""DataPilot — Semantic LLM cache for similar analytical questions.

Caches answers for sufficiently similar questions using embedding similarity.
Prevents incorrect cross-dataset cache reuse by scoping entries to datasets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("datapilot.cache.semantic")

# In-memory semantic cache store
# Structure: list of {dataset_id, question, embedding, answer, timestamp}
_semantic_store: list[dict[str, Any]] = []

_semantic_stats = {
    "hits": 0,
    "misses": 0,
    "avoided_llm_calls": 0,
    "tokens_saved_estimate": 0,
}


def _simple_text_embedding(text: str) -> np.ndarray:
    """Generate a simple TF-IDF-like embedding for a text string.
    
    Uses character n-grams and word hashing for a fast, dependency-free
    embedding that works well enough for question similarity.
    This avoids requiring an embedding API call for every cache lookup.
    """
    text = text.lower().strip()
    words = text.split()
    
    # Fixed-size feature vector
    dim = 256
    vec = np.zeros(dim, dtype=np.float32)
    
    # Word-level features
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vec[h] += 1.0
    
    # Character 3-gram features
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        h = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % dim
        vec[h] += 0.5
    
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class SemanticCache:
    """Semantic similarity cache for analytical questions.
    
    Scoped by dataset_id to prevent cross-dataset reuse.
    """

    def __init__(
        self,
        threshold: float = 0.92,
        max_entries: int = 500,
        ttl: int = 7200,
    ):
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl = ttl

    def get(self, question: str, dataset_id: str) -> Optional[dict[str, Any]]:
        """Look for a semantically similar cached answer.
        
        Returns the cached response dict if found, None otherwise.
        """
        query_emb = _simple_text_embedding(question)
        now = time.time()

        best_score = 0.0
        best_entry: Optional[dict] = None

        for entry in _semantic_store:
            # Must match dataset
            if entry["dataset_id"] != dataset_id:
                continue
            # Must not be expired
            if now - entry["timestamp"] > self.ttl:
                continue

            score = cosine_similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            _semantic_stats["hits"] += 1
            _semantic_stats["avoided_llm_calls"] += 1
            _semantic_stats["tokens_saved_estimate"] += best_entry.get("tokens_used", 500)
            logger.info(
                "Semantic cache HIT: score=%.3f q='%s' matched='%s'",
                best_score, question[:50], best_entry["question"][:50],
            )
            return {
                "answer": best_entry["answer"],
                "visualization": best_entry.get("visualization"),
                "chart_type": best_entry.get("chart_type"),
                "similarity_score": best_score,
                "original_question": best_entry["question"],
                "cached": True,
            }

        _semantic_stats["misses"] += 1
        logger.debug("Semantic cache MISS: q='%s' best_score=%.3f", question[:50], best_score)
        return None

    def set(
        self,
        question: str,
        dataset_id: str,
        answer: str,
        visualization: Optional[str] = None,
        chart_type: Optional[str] = None,
        tokens_used: int = 0,
    ) -> None:
        """Store an answer in the semantic cache."""
        embedding = _simple_text_embedding(question)

        entry = {
            "dataset_id": dataset_id,
            "question": question,
            "embedding": embedding,
            "answer": answer,
            "visualization": visualization,
            "chart_type": chart_type,
            "tokens_used": tokens_used,
            "timestamp": time.time(),
        }

        _semantic_store.append(entry)

        # Evict old entries if over capacity
        if len(_semantic_store) > self.max_entries:
            # Remove oldest entries
            _semantic_store.sort(key=lambda e: e["timestamp"])
            excess = len(_semantic_store) - self.max_entries
            del _semantic_store[:excess]

        logger.debug("Semantic cache SET: q='%s' dataset=%s", question[:50], dataset_id)

    def clear(self, dataset_id: Optional[str] = None) -> None:
        """Clear semantic cache entries, optionally for a specific dataset."""
        global _semantic_store
        if dataset_id:
            _semantic_store = [e for e in _semantic_store if e["dataset_id"] != dataset_id]
        else:
            _semantic_store.clear()

    @property
    def stats(self) -> dict[str, int]:
        return dict(_semantic_stats)

    @property
    def size(self) -> int:
        return len(_semantic_store)

    def reset_stats(self) -> None:
        _semantic_stats["hits"] = 0
        _semantic_stats["misses"] = 0
        _semantic_stats["avoided_llm_calls"] = 0
        _semantic_stats["tokens_saved_estimate"] = 0
