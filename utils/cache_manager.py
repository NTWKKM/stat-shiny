"""
Layer 1: Computation Caching System

In-memory computation cache with TTL (Time-To-Live) and LRU eviction.
Caches expensive statistical computation results to avoid redundant recalculations.

Features:
- Automatic expiration (TTL)
- LRU eviction when full
- Hash-based cache key generation
- Statistics tracking
- Thread-safe operations
"""

import hashlib
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from logger import get_logger

logger = get_logger(__name__)


class ComputationCache:
    """
    In-memory computation cache with TTL and LRU eviction.
    
    Caches expensive computation results (e.g., logistic regression analysis)
    to avoid redundant recalculation when parameters are unchanged.
    """
    
    def __init__(self, ttl_seconds: int = 1800, max_cache_size: int = 50):
        """
        Create a thread-safe in-memory computation cache configured with TTL and maximum size for LRU eviction.
        
        Parameters:
            ttl_seconds (int): Time-to-live for each cache entry in seconds; entries expire after this duration.
            max_cache_size (int): Maximum number of cached items before least-recently-used eviction occurs.
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds
        self.max_cache_size = max_cache_size
        self.access_count = {}  # Track access frequency for LRU
        self.hits = 0
        self.misses = 0
        logger.info(f"🟢 Cache initialized: TTL={ttl_seconds}s, MaxSize={max_cache_size}")
    
    def _make_key(self, func_name: str, kwargs: dict) -> str:
        """
        Create a deterministic cache key from a function name and its keyword arguments.
        
        Parameters:
            func_name (str): Identifier of the function to include in the key.
            kwargs (dict): Keyword arguments to include; values are serialized before hashing.
        
        Returns:
            key (str): MD5 hex digest of the JSON-serialized representation of {func_name: kwargs}.
        """
        key_str = json.dumps({func_name: kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, func_name: str, **kwargs) -> Optional[Any]:
        """
        Retrieve cached result if exists and not expired.
        
        Args:
            func_name: Function identifier
            **kwargs: Parameters used to create cache key
        
        Returns:
            Cached result or None if not found/expired
        """
        key = self._make_key(func_name, kwargs)
        
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if datetime.now() < entry['expires_at']:
                    self.access_count[key] = self.access_count.get(key, 0) + 1
                    self.hits += 1
                    logger.debug(f"✅ Cache HIT for {func_name} (key={key[:8]}...)")
                    return entry['result']
                else:
                    del self.cache[key]
                    if key in self.access_count:
                        del self.access_count[key]
                    logger.debug(f"⏰ Cache EXPIRED for {func_name}")
            
            self.misses += 1
            return None
    
    def set(self, func_name: str, result: Any, **kwargs) -> None:
        """
        Store a computation result in the cache with a time-to-live and LRU eviction.
        
        Inserts `result` under a deterministic key derived from `func_name` and `kwargs`, sets the entry's expiration to now + ttl, and initializes its access count to 1. If the cache has reached `max_cache_size`, evicts the least-recently-used entry before inserting. This operation acquires the cache's internal lock.
        
        Parameters:
            func_name (str): Identifier for the computation used when generating the cache key.
            result (Any): Value to store in the cache.
            **kwargs: Parameters that contribute to the cache key generation.
        """
        key = self._make_key(func_name, kwargs)
        
        with self._lock:
            # Implement simple LRU eviction
            if len(self.cache) >= self.max_cache_size:
                if not self.access_count:
                    # Fallback: remove arbitrary item if access_count is empty
                    evict_key = next(iter(self.cache.keys()))
                else:
                    # Remove least recently used item
                    evict_key = min(self.access_count.keys(), 
                                  key=lambda k: self.access_count.get(k, 0))
                
                del self.cache[evict_key]
                if evict_key in self.access_count:
                    del self.access_count[evict_key]
                logger.debug(f"♻️  Cache evicted LRU item (key={evict_key[:8]}...)")
                
            self.cache[key] = {
                'result': result,
                'expires_at': datetime.now() + timedelta(seconds=self.ttl),
                'created_at': datetime.now()
            }
            self.access_count[key] = 1
            logger.debug(f"💾 Cache SET for {func_name} (key={key[:8]}...)")
    
    def clear(self) -> None:
        """
        Remove all entries from the cache and reset cache statistics.
        
        Clears per-key access counts and resets hit and miss counters to zero.
        """
        with self._lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_count.clear()
            self.hits = 0
            self.misses = 0
            logger.info(f"🗑️  Cache cleared ({count} items removed)")
    
    def clear_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            int: The number of expired cache items that were removed.
        """
        with self._lock:
            expired_keys = [k for k, v in self.cache.items() 
                           if datetime.now() >= v['expires_at']]
            
            for k in expired_keys:
                del self.cache[k]
                if k in self.access_count:
                    del self.access_count[k]
            
            if expired_keys:
                logger.debug(f"🧹 Cache cleanup: removed {len(expired_keys)} expired items")
            
            return len(expired_keys)
    
    def get_stats(self) -> dict:
        """
        Return a snapshot of cache performance and configuration metrics.
        
        Returns:
            dict: Mapping of metrics:
                - cached_items (int): Number of entries currently stored in the cache.
                - max_size (int): Configured maximum number of cache entries.
                - hits (int): Number of cache hits.
                - misses (int): Number of cache misses.
                - hit_rate (str): Hit rate formatted as a percentage string (e.g. "75.0%").
                - ttl_seconds (int): Time-to-live for cache entries in seconds.
                - total_requests (int): Sum of hits and misses.
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cached_items': len(self.cache),
                'max_size': self.max_cache_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'ttl_seconds': self.ttl,
                'total_requests': total_requests
            }
    
    def __repr__(self) -> str:
        """
        Provide a compact string describing the cache's current item count and hit rate.
        
        Returns:
            str: A string in the form "ComputationCache(cached=X/Y, hit_rate=Z)" where X is the current number of cached items, Y is the maximum cache size, and Z is the hit rate percentage.
        """
        stats = self.get_stats()
        return f"ComputationCache(cached={stats['cached_items']}/{stats['max_size']}, hit_rate={stats['hit_rate']})"


# Global cache instance
COMPUTATION_CACHE = ComputationCache(ttl_seconds=1800, max_cache_size=50)