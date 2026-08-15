"""DataPilot — Redis exact cache for deterministic tool results.

Caches repeatable MCP operations (schema, filters, aggregations, queries)
with hash-based keys. Falls back to in-memory dict when Redis is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("datapilot.cache")

# In-memory fallback store
_memory_cache: dict[str, tuple[Any, float]] = {}  # key → (value, expire_timestamp)
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "avoided_tool_calls": 0,
}


class RedisCache:
    """Exact-match cache using Upstash Redis with in-memory fallback."""

    def __init__(self, redis_url: Optional[str] = None, redis_token: Optional[str] = None, ttl: int = 3600):
        self.ttl = ttl
        self._redis = None
        self._using_redis = False

        if redis_url and redis_token:
            try:
                try:
                    from upstash_redis import Redis  # type: ignore
                except ImportError:
                    Redis = None

                if Redis is not None:
                    self._redis = Redis(url=redis_url, token=redis_token)
                    # Test connection
                    self._redis.ping()
                    self._using_redis = True
                    logger.info("Redis cache connected: %s", redis_url[:40])
                else:
                    logger.info("upstash_redis not installed, using in-memory cache")
            except Exception as e:
                logger.warning("Redis unavailable, using in-memory fallback: %s", e)
                self._redis = None
                self._using_redis = False
        else:
            logger.info("No Redis credentials provided, using in-memory cache")

    @staticmethod
    def _make_key(tool_name: str, params: dict[str, Any]) -> str:
        """Create a deterministic cache key from tool name + parameters."""
        # Sort params for deterministic hashing
        param_str = json.dumps(params, sort_keys=True, default=str)
        raw = f"{tool_name}:{param_str}"
        return f"dp:tool:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, tool_name: str, params: dict[str, Any]) -> Optional[Any]:
        """Look up a cached tool result. Returns None on miss."""
        key = self._make_key(tool_name, params)

        if self._using_redis:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    _cache_stats["hits"] += 1
                    _cache_stats["avoided_tool_calls"] += 1
                    logger.debug("Cache HIT: %s(%s)", tool_name, params)
                    return json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                logger.warning("Redis GET failed: %s", e)
        else:
            # In-memory fallback
            if key in _memory_cache:
                value, expires = _memory_cache[key]
                if time.time() < expires:
                    _cache_stats["hits"] += 1
                    _cache_stats["avoided_tool_calls"] += 1
                    logger.debug("Memory cache HIT: %s(%s)", tool_name, params)
                    return value
                else:
                    del _memory_cache[key]

        _cache_stats["misses"] += 1
        logger.debug("Cache MISS: %s(%s)", tool_name, params)
        return None

    def set(self, tool_name: str, params: dict[str, Any], value: Any) -> None:
        """Store a tool result in cache."""
        key = self._make_key(tool_name, params)
        serialized = json.dumps(value, default=str)

        if self._using_redis:
            try:
                self._redis.setex(key, self.ttl, serialized)
                logger.debug("Cached: %s(%s) TTL=%ds", tool_name, params, self.ttl)
            except Exception as e:
                logger.warning("Redis SET failed, falling back to memory: %s", e)
                _memory_cache[key] = (value, time.time() + self.ttl)
        else:
            _memory_cache[key] = (value, time.time() + self.ttl)

    def invalidate(self, tool_name: str, params: dict[str, Any]) -> None:
        """Remove a specific entry from cache."""
        key = self._make_key(tool_name, params)
        if self._using_redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        if key in _memory_cache:
            del _memory_cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        _memory_cache.clear()
        if self._using_redis:
            try:
                # Clear only our namespace
                keys = self._redis.keys("dp:tool:*")
                if keys:
                    self._redis.delete(*keys)
            except Exception as e:
                logger.warning("Redis clear failed: %s", e)

    @property
    def stats(self) -> dict[str, int]:
        return dict(_cache_stats)

    @property
    def is_redis(self) -> bool:
        return self._using_redis

    def reset_stats(self) -> None:
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0
        _cache_stats["avoided_tool_calls"] = 0
