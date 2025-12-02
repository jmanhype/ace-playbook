"""
Two-level caching system (L1 in-memory + L2 Redis).

Provides TieredCache class with:
- L1: Fast in-memory LRU cache (local, per-process)
- L2: Distributed Redis cache (shared, persistent)
- Automatic fallback: L1 miss → L2 lookup → backfill L1
- Coordinated invalidation across both levels

Usage:
    from umes.utils.cache import TieredCache

    # Initialize cache
    cache = TieredCache(
        max_l1_size=1000,
        redis_url="redis://localhost:6379/0",
        default_ttl=3600
    )

    # Connect to Redis (optional, can work in L1-only mode)
    await cache.connect()

    # Cache operations
    await cache.set("user:123", {"name": "Alice"})
    user = await cache.get("user:123")
    await cache.delete("user:123")

    # Cleanup
    await cache.close()

Architecture:
    - L1 uses Python dict with manual LRU eviction (collections.OrderedDict)
    - L2 uses Redis with JSON serialization
    - L1 miss triggers L2 lookup and backfills L1 on hit
    - set() writes to both levels
    - delete()/clear() invalidates both levels
"""
import json
import asyncio
from typing import Any, Optional, Dict
from collections import OrderedDict

import redis.asyncio as redis


class TieredCache:
    """Two-level cache with in-memory L1 and Redis L2.

    Attributes:
        max_l1_size: Maximum number of entries in L1 cache
        redis_url: Redis connection URL (optional, L1-only if None)
        default_ttl: Default TTL in seconds for L2 entries (None = no expiry)
    """

    def __init__(
        self,
        max_l1_size: int = 1000,
        redis_url: Optional[str] = None,
        default_ttl: Optional[int] = None,
    ):
        """Initialize tiered cache.

        Args:
            max_l1_size: Maximum entries in L1 cache (LRU eviction)
            redis_url: Redis URL (e.g., "redis://localhost:6379/0")
            default_ttl: Default TTL in seconds for L2 (None = no expiry)
        """
        self.max_l1_size = max_l1_size
        self.redis_url = redis_url
        self.default_ttl = default_ttl

        # L1: In-memory LRU cache (OrderedDict maintains insertion order)
        self._l1_cache: OrderedDict[str, Any] = OrderedDict()

        # L2: Redis client (initialized in connect())
        self._redis_client: Optional[redis.Redis] = None

        # Cache key prefix for namespacing
        self._key_prefix = "umes:cache:"

    async def connect(self) -> None:
        """Establish Redis connection for L2 cache.

        Call this during application startup if using Redis.
        Optional - cache works in L1-only mode without calling this.
        """
        if self.redis_url:
            self._redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

    async def close(self) -> None:
        """Close Redis connection.

        Call this during application shutdown to release resources.
        """
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache.

        Lookup order:
        1. Check L1 (in-memory) - fast path
        2. If L1 miss, check L2 (Redis)
        3. If L2 hit, backfill L1 and return value
        4. If both miss, return None

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Try L1 first
        if key in self._l1_cache:
            # Move to end to mark as recently used (LRU)
            self._l1_cache.move_to_end(key)
            return self._l1_cache[key]

        # L1 miss - try L2
        l2_value = await self._get_l2(key)
        if l2_value is not None:
            # L2 hit - backfill L1
            self._set_l1(key, l2_value)
            return l2_value

        # Both miss
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store value in cache (both L1 and L2).

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable for L2)
            ttl_seconds: TTL for L2 in seconds (None = use default_ttl)
        """
        # Write to L1
        self._set_l1(key, value)

        # Write to L2 (if connected)
        if self._redis_client:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            await self._set_l2(key, value, ttl)

    async def delete(self, key: str) -> None:
        """Remove key from cache (both L1 and L2).

        Args:
            key: Cache key to delete
        """
        # Remove from L1
        self._l1_cache.pop(key, None)

        # Remove from L2
        if self._redis_client:
            await self._redis_client.delete(self._make_redis_key(key))

    async def clear(self) -> None:
        """Clear all entries from cache (both L1 and L2).

        Note: L2 clear removes only keys with our prefix to avoid
        affecting other Redis data.
        """
        # Clear L1
        self._l1_cache.clear()

        # Clear L2 (only our keys)
        if self._redis_client:
            pattern = f"{self._key_prefix}*"
            async for key in self._redis_client.scan_iter(match=pattern):
                await self._redis_client.delete(key)

    def _set_l1(self, key: str, value: Any) -> None:
        """Store value in L1 cache with LRU eviction.

        Args:
            key: Cache key
            value: Value to store
        """
        # If key exists, remove it first (will re-add at end)
        if key in self._l1_cache:
            del self._l1_cache[key]

        # Add to end (most recently used)
        self._l1_cache[key] = value

        # Evict oldest if over capacity
        if len(self._l1_cache) > self.max_l1_size:
            # popitem(last=False) removes oldest (FIFO/LRU)
            self._l1_cache.popitem(last=False)

    async def _get_l2(self, key: str) -> Optional[Any]:
        """Retrieve value from L2 (Redis) cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self._redis_client:
            return None

        try:
            redis_key = self._make_redis_key(key)
            serialized = await self._redis_client.get(redis_key)

            if serialized is None:
                return None

            # Deserialize JSON
            return json.loads(serialized)
        except Exception:
            # Redis error - return None (degrade gracefully)
            return None

    async def _set_l2(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int],
    ) -> None:
        """Store value in L2 (Redis) cache.

        Args:
            key: Cache key
            value: Value to store (must be JSON-serializable)
            ttl_seconds: TTL in seconds (None = no expiry)
        """
        if not self._redis_client:
            return

        try:
            redis_key = self._make_redis_key(key)
            serialized = json.dumps(value)

            if ttl_seconds:
                await self._redis_client.setex(
                    redis_key,
                    ttl_seconds,
                    serialized,
                )
            else:
                await self._redis_client.set(redis_key, serialized)
        except Exception:
            # Redis error - silently fail (degrade to L1-only)
            pass

    def _make_redis_key(self, key: str) -> str:
        """Generate namespaced Redis key.

        Args:
            key: Application cache key

        Returns:
            Redis key with prefix
        """
        return f"{self._key_prefix}{key}"
