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
    Retrieve a previously cached survival computation or compute and cache the result if absent.
    
    Parameters:
        calculate_func (callable): Zero-argument callable that computes and returns the desired result when a cache miss occurs.
        cache_key_params (dict): Parameters used to build the cache key for lookup and storage.
    
    Returns:
        The cached value if present for the given cache key, otherwise the value returned by `calculate_func` after it has been cached.
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
    Retrieve Kaplan–Meier curves from the layer-1 cache, computing and caching them if absent.
    
    Parameters:
        calculate_func (callable): Function that computes the KM curves when a cache miss occurs.
        cache_key_params (dict): Parameters used to form the cache key identifying the computation.
    
    Returns:
        The Kaplan–Meier curve result (the cached value if available, otherwise the newly computed result).
    """
    return _get_cached_result('survival_km', 'KM', calculate_func, cache_key_params)


def get_cached_na_curves(calculate_func, cache_key_params: dict):
    """
    Retrieve Nelson–Aalen cumulative hazard curves from the layer-1 cache or compute and cache them if not present.
    
    Parameters:
        calculate_func (callable): A callable that computes and returns Nelson–Aalen curve data when invoked.
        cache_key_params (dict): Parameters used to construct the cache key for lookup and invalidation.
    
    Returns:
        The Nelson–Aalen curve data produced by `calculate_func` or retrieved from cache.
    """
    return _get_cached_result('survival_na', 'NA', calculate_func, cache_key_params)


def get_cached_cox_model(calculate_func, cache_key_params: dict):
    """
    Retrieve a cached Cox proportional hazards model or compute and cache it if absent.
    
    Parameters:
        calculate_func (callable): Function that fits and returns a Cox model when called.
        cache_key_params (dict): Parameters used to build the cache key identifying the model.
    
    Returns:
        Cox model object from cache or a newly fitted Cox model.
    """
    return _get_cached_result('survival_cox', 'Cox', calculate_func, cache_key_params)


def get_cached_survival_estimates(calculate_func, cache_key_params: dict):
    """
    Retrieve cached survival estimates for the given cache key or compute and cache them if absent.
    
    Parameters:
        calculate_func (callable): Function invoked to compute survival estimates on a cache miss.
        cache_key_params (dict): Parameters used to construct the cache key for lookup and storage.
    
    Returns:
        The survival estimates object returned by `calculate_func` or retrieved from the cache.
    """
    return _get_cached_result('survival_estimates', 'Estimates', calculate_func, cache_key_params)


def get_cached_risk_table(calculate_func, cache_key_params: dict):
    """
    Retrieve a risk table from the layer-1 cache or compute and cache it if absent.
    
    Parameters:
        calculate_func: Callable that produces the risk table when a cached value is missing.
        cache_key_params (dict): Parameters used to form the cache key for lookup and invalidation.
    
    Returns:
        The risk table object or data structure returned by `calculate_func`, sourced from cache when available.
    """
    return _get_cached_result('survival_risk_table', 'Risk Table', calculate_func, cache_key_params)