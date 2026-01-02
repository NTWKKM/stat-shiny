"""
Layer 1 Cache Integration for PSM (Propensity Score Matching)

Caches propensity score calculations to avoid redundant computation.
Propensity scores are expensive to calculate and often requested with same parameters.

Features:
- Cache propensity scores for 30 minutes
- LRU eviction when cache full
- Automatic hash-based invalidation
- Thread-safe operations
"""

from utils.cache_manager import COMPUTATION_CACHE
from logger import get_logger

logger = get_logger(__name__)


def _get_cached_result(cache_namespace: str, calculate_func, cache_key_params: dict, operation_name: str):
    """
    Generic cache helper for PSM computations with error handling.
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get(cache_namespace, **cache_key_params)
    if cached is not None:
        logger.info(f"✅ {operation_name} Cache HIT - using cached results")
        return cached
    
    logger.info(f"📊 {operation_name} Cache MISS - performing calculation")
    
    try:
        # Calculate and cache
        result = calculate_func()
        COMPUTATION_CACHE.set(cache_namespace, result, **cache_key_params)
        logger.info(f"💾 {operation_name} result cached for 30 minutes")
        return result
    except Exception as e:
        logger.error(f"❌ {operation_name} calculation failed: {str(e)}")
        raise


def get_cached_propensity_scores(calculate_func, cache_key_params: dict):
    """
    Get propensity scores from cache or calculate if not cached.
    
    Args:
        calculate_func: Function that calculates propensity scores
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Propensity scores (from cache or fresh calculation)
    """
    return _get_cached_result('psm_calculate', calculate_func, cache_key_params, 'PSM Score')


def get_cached_matched_data(matching_func, cache_key_params: dict):
    """
    Get matched dataset from cache or perform matching if not cached.
    
    Args:
        matching_func: Function that performs matching
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Matched dataframe (from cache or fresh matching)
    """
    return _get_cached_result('psm_matching', matching_func, cache_key_params, 'PSM Matching')


def get_cached_smd(smd_func, cache_key_params: dict):
    """
    Get SMD calculations from cache or perform calculation if not cached.
    
    Args:
        smd_func: Function that calculates SMD
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        SMD dataframe (from cache or fresh calculation)
    """
    return _get_cached_result('psm_smd', smd_func, cache_key_params, 'PSM SMD')
