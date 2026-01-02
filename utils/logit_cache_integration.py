"""
Layer 1 Cache Integration for Logistic Regression

Caches logistic regression analysis results to avoid redundant computation.
Logistic regression (especially multivariate with VIF checks) can be computationally intensive.

Features:
- Cache outcome analysis results (HTML tables, OR/aOR dicts)
- LRU eviction
- Hash-based invalidation
"""

from utils.cache_manager import COMPUTATION_CACHE
from logger import get_logger
from typing import Callable, Any, Tuple, Optional

logger = get_logger(__name__)

def get_cached_logistic_analysis(
    calculate_func: Callable[[], Tuple[Any, Any, Any]], 
    cache_key_params: dict
) -> Optional[Tuple[Any, Any, Any]]:
    """
    Retrieve cached logistic regression analysis results or compute and cache them.
    
    Parameters:
        calculate_func (Callable[[], Tuple[Any, Any, Any]]): Function that performs the analysis and returns (html_table, or_results, aor_results).
        cache_key_params (dict): Parameters used to form the cache key for storing/retrieving the analysis result.
    
    Returns:
        Optional[Tuple[Any, Any, Any]]: The tuple (html_table, or_results, aor_results) returned from cache or from `calculate_func()`, or `None` if the calculation produced a falsy result.
    
    Notes:
        Only a truthy 3-tuple is stored in the cache; invalid or falsy results are returned but not cached.
    """
    # Try cache first
    cached = COMPUTATION_CACHE.get('analyze_outcome', **cache_key_params)
    if cached is not None:
        logger.info("✅ Logit Analysis Cache HIT - using cached results")
        return cached
    
    logger.info("📊 Logit Analysis Cache MISS - performing analysis")
    
    # Calculate and cache
    result = calculate_func()
    
    # Validation before caching: Check if result is valid
    # result is (html_table, or_results, aor_results)
    if result and isinstance(result, tuple) and len(result) == 3:
        COMPUTATION_CACHE.set('analyze_outcome', result, **cache_key_params)
        logger.info("💾 Logit analysis results cached for 30 minutes")
    else:
        logger.warning("Logit analysis returned invalid result, not caching")
    
    return result