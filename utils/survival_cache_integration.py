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


def get_cached_km_curves(calculate_func, cache_key_params: dict):
    """
    Get Kaplan-Meier curves from cache or calculate if not cached.
    
    Args:
        calculate_func: Function that calculates KM curves
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        KM curve data (from cache or fresh calculation)
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('survival_km', **cache_key_params)
    if cached is not None:
        logger.info(f"✅ Survival KM Cache HIT - using cached curves")
        return cached
    
    logger.info(f"⏳ Survival KM Cache MISS - calculating Kaplan-Meier curves")
    
    # Calculate and cache
    result = calculate_func()
    COMPUTATION_CACHE.set('survival_km', result, **cache_key_params)
    logger.info(f"💾 Survival KM curves cached for 30 minutes")
    
    return result

def get_cached_na_curves(calculate_func, cache_key_params: dict):
    """
    Get Nelson-Aalen curves from cache or calculate if not cached.
    
    Args:
        calculate_func: Function that calculates NA curves
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        NA curve data (from cache or fresh calculation)
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('survival_na', **cache_key_params)
    if cached is not None:
        logger.info(f"✅ Survival NA Cache HIT - using cached curves")
        return cached
    
    logger.info(f"⏳ Survival NA Cache MISS - calculating Nelson-Aalen curves")
    
    # Calculate and cache
    result = calculate_func()
    COMPUTATION_CACHE.set('survival_na', result, **cache_key_params)
    logger.info(f"💾 Survival NA curves cached for 30 minutes")
    
    return result


def get_cached_cox_model(calculate_func, cache_key_params: dict):
    """
    Get Cox regression model from cache or fit if not cached.
    
    Args:
        calculate_func: Function that fits Cox model
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Cox model object (from cache or fresh fit)
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('survival_cox', **cache_key_params)
    if cached is not None:
        logger.info(f"✅ Survival Cox Cache HIT - using cached model")
        return cached
    
    logger.info(f"🔄 Survival Cox Cache MISS - fitting Cox model")
    
    # Fit and cache
    result = calculate_func()
    COMPUTATION_CACHE.set('survival_cox', result, **cache_key_params)
    logger.info(f"💾 Survival Cox model cached for 30 minutes")
    
    return result


def get_cached_survival_estimates(calculate_func, cache_key_params: dict):
    """
    Get survival estimates from cache or calculate if not cached.
    
    Args:
        calculate_func: Function that calculates survival estimates
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Survival estimates (from cache or fresh calculation)
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('survival_estimates', **cache_key_params)
    if cached is not None:
        logger.info(f"✅ Survival Estimates Cache HIT - using cached estimates")
        return cached
    
    logger.info(f"💋 Survival Estimates Cache MISS - calculating estimates")
    
    # Calculate and cache
    result = calculate_func()
    COMPUTATION_CACHE.set('survival_estimates', result, **cache_key_params)
    logger.info(f"💾 Survival estimates cached for 30 minutes")
    
    return result


def get_cached_risk_table(calculate_func, cache_key_params: dict):
    """
    Get risk table from cache or generate if not cached.
    
    Args:
        calculate_func: Function that generates risk table
        cache_key_params: Dict with parameters for cache key
    
    Returns:
        Risk table data (from cache or fresh generation)
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('survival_risk_table', **cache_key_params)
    if cached is not None:
        logger.info(f"✅ Survival Risk Table Cache HIT - using cached table")
        return cached
    
    logger.info(f"📄 Survival Risk Table Cache MISS - generating table")
    
    # Generate and cache
    result = calculate_func()
    COMPUTATION_CACHE.set('survival_risk_table', result, **cache_key_params)
    logger.info(f"💾 Survival risk table cached for 30 minutes")
    
    return result
