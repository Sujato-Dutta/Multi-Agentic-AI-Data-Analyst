"""Tests for Redis exact cache and semantic cache."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.cache.redis_cache import RedisCache
from app.cache.semantic_cache import SemanticCache, cosine_similarity, _simple_text_embedding


class TestRedisCache:
    """Tests for the exact/deterministic cache (in-memory fallback)."""

    def setup_method(self):
        self.cache = RedisCache(redis_url=None, redis_token=None, ttl=60)
        self.cache.clear()
        self.cache.reset_stats()

    def test_set_and_get(self):
        self.cache.set("get_schema", {"dataset_id": "test"}, {"columns": ["a", "b"]})
        result = self.cache.get("get_schema", {"dataset_id": "test"})
        assert result is not None
        assert result["columns"] == ["a", "b"]

    def test_cache_miss(self):
        result = self.cache.get("get_schema", {"dataset_id": "nonexistent"})
        assert result is None

    def test_cache_hit_stats(self):
        self.cache.set("tool", {"k": "v"}, {"result": 1})
        self.cache.get("tool", {"k": "v"})
        assert self.cache.stats["hits"] == 1

    def test_cache_miss_stats(self):
        self.cache.get("tool", {"k": "missing"})
        assert self.cache.stats["misses"] == 1

    def test_deterministic_keys(self):
        """Same tool+params should produce same cache key."""
        self.cache.set("filter", {"col": "a", "val": 1}, {"data": [1, 2]})
        # Same params in different order should hit
        result = self.cache.get("filter", {"val": 1, "col": "a"})
        assert result is not None

    def test_different_params_miss(self):
        self.cache.set("filter", {"col": "a"}, {"data": [1]})
        result = self.cache.get("filter", {"col": "b"})
        assert result is None

    def test_invalidate(self):
        self.cache.set("tool", {"k": "v"}, {"result": 1})
        self.cache.invalidate("tool", {"k": "v"})
        result = self.cache.get("tool", {"k": "v"})
        assert result is None

    def test_clear(self):
        self.cache.set("t1", {"k": "1"}, {"r": 1})
        self.cache.set("t2", {"k": "2"}, {"r": 2})
        self.cache.clear()
        assert self.cache.get("t1", {"k": "1"}) is None
        assert self.cache.get("t2", {"k": "2"}) is None

    def test_is_not_redis(self):
        assert self.cache.is_redis is False


class TestSemanticCache:
    """Tests for the semantic similarity cache."""

    def setup_method(self):
        self.cache = SemanticCache(threshold=0.85, max_entries=100, ttl=3600)
        self.cache.clear()
        self.cache.reset_stats()

    def test_set_and_get_exact(self):
        self.cache.set("What is the total revenue?", "ds1", "The total revenue is $100,000")
        result = self.cache.get("What is the total revenue?", "ds1")
        assert result is not None
        assert "100,000" in result["answer"]

    def test_similar_question_hit(self):
        self.cache.set("What is the total revenue?", "ds1", "$100K")
        result = self.cache.get("What is total revenue?", "ds1")
        # Very similar question should hit
        assert result is not None

    def test_different_question_miss(self):
        self.cache.set("What is the total revenue?", "ds1", "$100K")
        result = self.cache.get("Which region has the most customers?", "ds1")
        # Very different question should miss
        assert result is None

    def test_cross_dataset_isolation(self):
        """Entries from one dataset should NOT match another."""
        self.cache.set("What is the total revenue?", "dataset_A", "Answer for A")
        result = self.cache.get("What is the total revenue?", "dataset_B")
        assert result is None

    def test_stats_tracking(self):
        self.cache.set("Q1", "ds1", "A1")
        self.cache.get("Q1", "ds1")  # hit
        self.cache.get("Totally different question", "ds1")  # miss
        assert self.cache.stats["hits"] >= 1
        assert self.cache.stats["misses"] >= 1

    def test_max_entries_eviction(self):
        cache = SemanticCache(threshold=0.85, max_entries=5, ttl=3600)
        for i in range(10):
            cache.set(f"Question number {i} about topic {i}", "ds1", f"Answer {i}")
        assert cache.size <= 5

    def test_clear_specific_dataset(self):
        self.cache.set("Q1", "ds1", "A1")
        self.cache.set("Q2", "ds2", "A2")
        self.cache.clear(dataset_id="ds1")
        assert self.cache.get("Q1", "ds1") is None
        # ds2 should still be there
        result = self.cache.get("Q2", "ds2")
        assert result is not None


class TestEmbeddings:
    """Tests for the simple text embedding function."""

    def test_same_text_same_embedding(self):
        e1 = _simple_text_embedding("hello world")
        e2 = _simple_text_embedding("hello world")
        assert cosine_similarity(e1, e2) == pytest.approx(1.0, abs=1e-6)

    def test_similar_text_high_similarity(self):
        e1 = _simple_text_embedding("what is the total revenue")
        e2 = _simple_text_embedding("what is total revenue")
        assert cosine_similarity(e1, e2) > 0.8

    def test_different_text_low_similarity(self):
        e1 = _simple_text_embedding("what is the total revenue")
        e2 = _simple_text_embedding("show me all customers in north region")
        assert cosine_similarity(e1, e2) < 0.7

    def test_embedding_normalized(self):
        import numpy as np
        e = _simple_text_embedding("test query")
        norm = np.linalg.norm(e)
        assert norm == pytest.approx(1.0, abs=1e-5)
