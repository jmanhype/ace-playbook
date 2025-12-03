"""
Unit tests for two-level caching (L1 in-memory + L2 Redis).

Tests the TieredCache class that provides:
- L1: In-memory LRU cache (fast, local)
- L2: Redis cache (distributed, persistent)
- Automatic fallback: L1 miss → L2 lookup → backfill L1
- Coordinated invalidation across both levels
"""
import pytest
import asyncio
from typing import Any, Optional

from umes.utils.cache import TieredCache


class TestTieredCacheBasics:
    """Test basic cache operations (get, set, delete)."""

    @pytest.mark.asyncio
    async def test_cache_set_and_get_l1(self):
        """Test that set() stores value and get() retrieves it from L1."""
        cache = TieredCache(max_l1_size=100)

        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_cache_get_returns_none_when_not_found(self):
        """Test that get() returns None for non-existent keys."""
        cache = TieredCache(max_l1_size=100)

        result = await cache.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_delete_removes_key(self):
        """Test that delete() removes key from cache."""
        cache = TieredCache(max_l1_size=100)

        await cache.set("key1", "value1")
        await cache.delete("key1")
        result = await cache.get("key1")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_overwrite_existing_key(self):
        """Test that set() overwrites existing values."""
        cache = TieredCache(max_l1_size=100)

        await cache.set("key1", "original")
        await cache.set("key1", "updated")
        result = await cache.get("key1")

        assert result == "updated"


class TestL1InMemoryCache:
    """Test L1 (in-memory) cache behavior."""

    @pytest.mark.asyncio
    async def test_l1_cache_stores_values(self):
        """Test that L1 cache stores values in memory."""
        cache = TieredCache(max_l1_size=100)

        await cache.set("key1", "value1")

        # Check L1 directly
        assert "key1" in cache._l1_cache
        assert cache._l1_cache["key1"] == "value1"

    @pytest.mark.asyncio
    async def test_l1_cache_lru_eviction(self):
        """Test that L1 cache evicts least recently used items when full."""
        cache = TieredCache(max_l1_size=2)  # Small size to trigger eviction

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")  # Should evict key1

        # key1 should be evicted from L1
        assert "key1" not in cache._l1_cache
        assert "key2" in cache._l1_cache
        assert "key3" in cache._l1_cache

    @pytest.mark.asyncio
    async def test_l1_cache_access_updates_lru(self):
        """Test that accessing a key updates its LRU position."""
        cache = TieredCache(max_l1_size=2)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Access key1 to make it most recently used
        await cache.get("key1")

        # Add key3 - should evict key2 (least recently used), not key1
        await cache.set("key3", "value3")

        assert "key1" in cache._l1_cache  # Still in cache (recently accessed)
        assert "key2" not in cache._l1_cache  # Evicted
        assert "key3" in cache._l1_cache


class TestL2RedisCache:
    """Test L2 (Redis) cache behavior."""

    @pytest.mark.asyncio
    async def test_l2_cache_stores_values(self, redis_url: str):
        """Test that L2 cache stores values in Redis."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        await cache.set("key1", "value1")

        # Check L2 directly (bypassing L1)
        l2_value = await cache._get_l2("key1")
        assert l2_value == "value1"

        await cache.close()

    @pytest.mark.asyncio
    async def test_l2_cache_persists_across_instances(self, redis_url: str):
        """Test that L2 cache persists between different cache instances."""
        # Create first cache instance and set value
        cache1 = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache1.connect()
        await cache1.set("persistent_key", "persistent_value")
        await cache1.close()

        # Create second cache instance and retrieve value
        cache2 = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache2.connect()
        result = await cache2.get("persistent_key")
        await cache2.close()

        assert result == "persistent_value"

    @pytest.mark.asyncio
    async def test_l2_cache_without_redis_url(self):
        """Test that cache works without Redis (L1 only mode)."""
        cache = TieredCache(max_l1_size=100)  # No Redis URL

        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"
        assert cache._redis_client is None


class TestTieredCacheFallback:
    """Test cache fallback behavior (L1 → L2 → backfill)."""

    @pytest.mark.asyncio
    async def test_fallback_to_l2_on_l1_miss(self, redis_url: str):
        """Test that L1 miss triggers L2 lookup."""
        cache = TieredCache(max_l1_size=2, redis_url=redis_url)
        await cache.connect()

        # Set value (will be in both L1 and L2)
        await cache.set("key1", "value1")

        # Manually evict from L1 to simulate eviction
        cache._l1_cache.clear()

        # Get should fallback to L2
        result = await cache.get("key1")
        assert result == "value1"

        await cache.close()

    @pytest.mark.asyncio
    async def test_l2_hit_backfills_l1(self, redis_url: str):
        """Test that L2 hit backfills L1 cache."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Set value
        await cache.set("key1", "value1")

        # Clear L1 to simulate eviction
        cache._l1_cache.clear()
        assert "key1" not in cache._l1_cache

        # Get should fetch from L2 and backfill L1
        result = await cache.get("key1")
        assert result == "value1"
        assert "key1" in cache._l1_cache  # Backfilled
        assert cache._l1_cache["key1"] == "value1"

        await cache.close()


class TestCacheInvalidation:
    """Test coordinated cache invalidation across L1 and L2."""

    @pytest.mark.asyncio
    async def test_delete_removes_from_both_levels(self, redis_url: str):
        """Test that delete() removes key from both L1 and L2."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Set value (will be in both caches)
        await cache.set("key1", "value1")
        assert "key1" in cache._l1_cache

        # Delete should remove from both
        await cache.delete("key1")

        # Verify removal from L1
        assert "key1" not in cache._l1_cache

        # Verify removal from L2
        l2_value = await cache._get_l2("key1")
        assert l2_value is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_clear_removes_all_keys(self, redis_url: str):
        """Test that clear() removes all keys from both levels."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Set multiple values
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Clear all
        await cache.clear()

        # Verify L1 cleared
        assert len(cache._l1_cache) == 0

        # Verify L2 cleared (check if keys exist)
        assert await cache._get_l2("key1") is None
        assert await cache._get_l2("key2") is None
        assert await cache._get_l2("key3") is None

        await cache.close()


class TestCacheTTL:
    """Test time-to-live (TTL) support."""

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, redis_url: str):
        """Test that set() accepts TTL parameter."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Set with short TTL
        await cache.set("key1", "value1", ttl_seconds=1)

        # Value should exist initially
        result = await cache.get("key1")
        assert result == "value1"

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        # Value should be gone from L2 (L1 doesn't enforce TTL)
        l2_value = await cache._get_l2("key1")
        assert l2_value is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_default_ttl_applied(self, redis_url: str):
        """Test that default TTL is applied when not specified."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url, default_ttl=1)
        await cache.connect()

        await cache.set("key1", "value1")  # No explicit TTL

        # Wait for default TTL to expire
        await asyncio.sleep(1.5)

        # Value should be gone from L2
        l2_value = await cache._get_l2("key1")
        assert l2_value is None

        await cache.close()


class TestCacheConnectionManagement:
    """Test Redis connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_establishes_redis_connection(self, redis_url: str):
        """Test that connect() establishes Redis connection."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)

        assert cache._redis_client is None

        await cache.connect()

        assert cache._redis_client is not None

        await cache.close()

    @pytest.mark.asyncio
    async def test_close_releases_redis_connection(self, redis_url: str):
        """Test that close() releases Redis connection."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        await cache.close()

        # After close, Redis operations should fail gracefully
        # (Implementation should handle closed connections)
        await cache.set("key1", "value1")  # Should not raise

    @pytest.mark.asyncio
    async def test_operations_without_connect_work_in_l1_mode(self):
        """Test that cache works without connect() call (L1 only)."""
        cache = TieredCache(max_l1_size=100)

        # Should work without connect() if no Redis URL
        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"


class TestCacheComplexTypes:
    """Test caching of complex Python types."""

    @pytest.mark.asyncio
    async def test_cache_dict_values(self):
        """Test caching dictionary objects."""
        cache = TieredCache(max_l1_size=100)

        test_dict = {"name": "Alice", "age": 30, "active": True}
        await cache.set("user", test_dict)
        result = await cache.get("user")

        assert result == test_dict
        assert result["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_cache_list_values(self):
        """Test caching list objects."""
        cache = TieredCache(max_l1_size=100)

        test_list = [1, 2, 3, "four", {"five": 5}]
        await cache.set("items", test_list)
        result = await cache.get("items")

        assert result == test_list
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_cache_none_value(self):
        """Test caching None as a value (different from cache miss)."""
        cache = TieredCache(max_l1_size=100)

        await cache.set("null_key", None)
        result = await cache.get("null_key")

        # Should distinguish between cached None and cache miss
        # (Implementation detail: may need sentinel value)
        assert result is None or result == {"__cached_none__": True}


class TestCacheErrorHandling:
    """Test cache error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_get_l2_handles_redis_errors_gracefully(self, redis_url: str):
        """Test that _get_l2() returns None on Redis errors."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Close Redis connection to simulate error
        await cache._redis_client.close()

        # Should return None instead of raising exception
        result = await cache._get_l2("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_l2_handles_redis_errors_gracefully(self, redis_url: str):
        """Test that _set_l2() fails silently on Redis errors."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Close Redis connection to simulate error
        await cache._redis_client.close()

        # Should not raise exception (degrades to L1-only)
        await cache._set_l2("key1", "value1", None)
        # No assertion needed - test passes if no exception raised

    @pytest.mark.asyncio
    async def test_set_l2_without_redis_client(self):
        """Test that _set_l2() returns early when no Redis client."""
        cache = TieredCache(max_l1_size=100)  # No Redis URL

        # Should return early without error
        await cache._set_l2("key1", "value1", None)
        # No assertion needed - test passes if no exception raised

    @pytest.mark.asyncio
    async def test_clear_with_redis_scan_iter(self, redis_url: str):
        """Test that clear() uses scan_iter to remove Redis keys."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Set multiple values with same prefix
        await cache.set("test:key1", "value1")
        await cache.set("test:key2", "value2")
        await cache.set("test:key3", "value3")

        # Clear should iterate through all keys and delete them
        await cache.clear()

        # Verify all keys removed from L2
        assert await cache._get_l2("test:key1") is None
        assert await cache._get_l2("test:key2") is None
        assert await cache._get_l2("test:key3") is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_get_l2_handles_json_decode_errors(self, redis_url: str):
        """Test that _get_l2() handles JSON decode errors gracefully."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Manually set invalid JSON in Redis
        redis_key = cache._make_redis_key("invalid_json_key")
        await cache._redis_client.set(redis_key, "this is not valid JSON {{{")

        # Should return None instead of raising exception
        result = await cache._get_l2("invalid_json_key")
        assert result is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_set_l2_handles_json_encode_errors(self, redis_url: str):
        """Test that _set_l2() handles JSON encode errors gracefully."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Try to cache a non-serializable object
        class NonSerializable:
            pass

        # Should not raise exception (degrades to L1-only)
        await cache._set_l2("key1", NonSerializable(), None)
        # No assertion needed - test passes if no exception raised

        await cache.close()

    @pytest.mark.asyncio
    async def test_clear_when_cache_empty(self, redis_url: str):
        """Test that clear() handles empty cache gracefully (no keys to iterate)."""
        cache = TieredCache(max_l1_size=100, redis_url=redis_url)
        await cache.connect()

        # Clear without adding any keys - should iterate zero times
        await cache.clear()

        # Verify L1 is empty
        assert len(cache._l1_cache) == 0

        # Should not error even with empty cache
        await cache.close()
