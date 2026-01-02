"""
Layer 1 Cache Integration for Survival Analysis

Caches survival model fits and survival estimates to avoid redundant computation.
Survival analysis is computationally expensive and often requested with same parameters.

Features:
- Cache Kaplan-Meier curves for 30 minutes
- Cache Nelson-Aalen curves for 30 minutes
- Cache Cox regression models for 30 minutes
- LRU eviction when cache full
- Automatic hash-based invalidation
- Thread-safe operations
"""

from utils.cache_manager import COMPUTATION_CACHE
from logger import get_logger

logger = get_logger(__name__)


def _get_cached_result(cache_type: str, label: str, calculate_func, cache_key_params: dict):
    """
    Generic cache wrapper for survival computations to reduce duplication.
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get(cache_type, **cache_key_params)
    if cached is not None:
        logger.info(f"✅ Survival {label} Cache HIT - using cached results")
        return cached
    
    logger.info(f"⏳ Survival {label} Cache MISS - computing")
    
    # Calculate and cache
    result = calculate_func()
    COMPUTATION_CACHE.set(cache_type, result, **cache_key_params)
    logger.info(f"💾 Survival {label} cached for 30 minutes")
    
    return result


def get_cached_km_curves(calculate_func, cache_key_params: dict):
    """
    Get Kaplan-Meier curves from cache or calculate if not cached.
    
    Args:
        calculate_func: Function that calculates KM curves
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        KM curve data (from cache or fresh calculation)
    """
    return _get_cached_result('survival_km', 'KM', calculate_func, cache_key_params)


def get_cached_na_curves(calculate_func, cache_key_params: dict):
    """
    Get Nelson-Aalen curves from cache or calculate if not cached.
    
    Args:
        calculate_func: Function that calculates NA curves
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        NA curve data (from cache or fresh calculation)
    """
    return _get_cached_result('survival_na', 'NA', calculate_func, cache_key_params)


def get_cached_cox_model(calculate_func, cache_key_params: dict):
    """
    Get Cox regression model from cache or fit if not cached.
    
    Args:
        calculate_func: Function that fits Cox model
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Cox model object (from cache or fresh fit)
    """
    return _get_cached_result('survival_cox', 'Cox', calculate_func, cache_key_params)


def get_cached_survival_estimates(calculate_func, cache_key_params: dict):
    """
    Get survival estimates from cache or calculate if not cached.
    
    Args:
        calculate_func: Function that calculates survival estimates
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Survival estimates (from cache or fresh calculation)
    """
    return _get_cached_result('survival_estimates', 'Estimates', calculate_func, cache_key_params)


def get_cached_risk_table(calculate_func, cache_key_params: dict):
    """
    Get risk table from cache or generate if not cached.
    
    Args:
        calculate_func: Function that generates risk table
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Risk table data (from cache or fresh generation)
    """
    return _get_cached_result('survival_risk_table', 'Risk Table', calculate_func, cache_key_params)
