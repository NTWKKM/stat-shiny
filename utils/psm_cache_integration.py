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


def get_cached_propensity_scores(calculate_func, cache_key_params: dict):
    """
    Get propensity scores from cache or calculate if not cached.
    
    Args:
        calculate_func: Function that calculates propensity scores
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Propensity scores (from cache or fresh calculation)
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('psm_calculate', **cache_key_params)
    if cached is not None:
        logger.info(f"✅ PSM Cache HIT - using cached propensity scores")
        return cached
    
    logger.info(f"📊 PSM Cache MISS - calculating propensity scores")
    
    # Calculate and cache
    result = calculate_func()
    COMPUTATION_CACHE.set('psm_calculate', result, **cache_key_params)
    logger.info(f"💾 PSM result cached for 30 minutes")
    
    return result


def get_cached_matched_data(matching_func, cache_key_params: dict):
    """
    Get matched dataset from cache or perform matching if not cached.
    
    Args:
        matching_func: Function that performs matching
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Matched dataframe (from cache or fresh matching)
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('psm_matching', **cache_key_params)
    if cached is not None:
        logger.info(f"✅ PSM Matching Cache HIT - using cached matched data")
        return cached
    
    logger.info(f"🔄 PSM Matching Cache MISS - performing matching")
    
    # Perform matching and cache
    result = matching_func()
    COMPUTATION_CACHE.set('psm_matching', result, **cache_key_params)
    logger.info(f"💾 PSM matching result cached for 30 minutes")
    
    return result


def get_cached_smd(smd_func, cache_key_params: dict):
    """
    Get SMD calculations from cache or perform calculation if not cached.
    
    Args:
        smd_func: Function that calculates SMD
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        SMD dataframe (from cache or fresh calculation)
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('psm_smd', **cache_key_params)
    if cached is not None:
        logger.info(f"✅ PSM SMD Cache HIT - using cached SMD results")
        return cached
    
    logger.info(f"📊 PSM SMD Cache MISS - calculating SMD")
    
    # Calculate and cache
    result = smd_func()
    COMPUTATION_CACHE.set('psm_smd', result, **cache_key_params)
    logger.info(f"💾 PSM SMD result cached for 30 minutes")
    
    return result
