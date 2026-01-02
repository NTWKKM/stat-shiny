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
    Retrieve a cached computation for the given namespace and key parameters, or run the provided callable to compute, cache, and return the result.
    
    Parameters:
        cache_namespace (str): Namespace within the computation cache where the result is stored.
        calculate_func (callable): Zero-argument callable invoked to produce the result when there is a cache miss.
        cache_key_params (dict): Parameters used to form the cache key that identifies the result.
        operation_name (str): Human-readable name used for logging messages.
    
    Returns:
        The cached value if present, otherwise the value returned by `calculate_func` after it is stored in the cache.
    
    Raises:
        Any exception raised by `calculate_func` is propagated after being logged.
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
    Retrieve propensity scores, using a cached result when available.
    
    Parameters:
        calculate_func (callable): Callable that computes propensity scores when a cache miss occurs.
        cache_key_params (dict): Parameters used to form the cache key (must uniquely identify the inputs for the calculation).
    
    Returns:
        The propensity scores produced by `calculate_func`.
    """
    return _get_cached_result('psm_calculate', calculate_func, cache_key_params, 'PSM Score')


def get_cached_matched_data(matching_func, cache_key_params: dict):
    """
    Retrieve the matched dataset from cache or compute and cache it when missing.
    
    Parameters:
        matching_func (callable): Callable that returns the matched dataset when invoked.
        cache_key_params (dict): Parameters used to construct the cache key for lookup/invalidation.
    
    Returns:
        matched_df: The matched dataset (typically a pandas DataFrame) obtained from cache or produced by `matching_func`.
    """
    return _get_cached_result('psm_matching', matching_func, cache_key_params, 'PSM Matching')


def get_cached_smd(smd_func, cache_key_params: dict):
    """
    Retrieve SMD results from the layer-1 cache or compute and cache them if missing.
    
    Parameters:
        smd_func (callable): Callable that computes and returns the SMD results (typically a pandas DataFrame).
        cache_key_params (dict): Parameters used to construct the cache key (e.g., dataset identifiers, feature/hash) that determine cache uniqueness.
    
    Returns:
        pandas.DataFrame: Standardized mean differences table, from cache if available or freshly computed.
    """
    return _get_cached_result('psm_smd', smd_func, cache_key_params, 'PSM SMD')