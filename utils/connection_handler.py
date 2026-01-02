"""
Layer 3: Connection Resilience System

Automatic retry logic with exponential backoff for network failures.
Prevents app crash from intermittent HF connection issues.

Features:
- Exponential backoff retry strategy
- Configurable retry attempts
- Network failure detection
- Graceful fallback on failure
"""

import time
import random
from typing import Callable, Any, Optional
from logger import get_logger

logger = get_logger(__name__)


class ConnectionHandler:
    """
    Handle network/connection issues with exponential backoff retry.
    """
    
    def __init__(self, max_retries: int = 3, initial_backoff: float = 0.5, backoff_factor: float = 2.0):
        """
        Initialize connection handler.
        
        Args:
            max_retries: Maximum retry attempts
            initial_backoff: Initial wait time in seconds
            backoff_factor: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.failed_attempts = 0
        self.successful_retries = 0
        self.total_calls = 0
        logger.info(f"🔌 Connection handler initialized: max_retries={max_retries}, initial_backoff={initial_backoff}s")
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """
        Execute function with exponential backoff retry on failure.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            Function result or None if all retries failed
        """
        self.total_calls += 1
        
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"✅ Succeeded after {attempt + 1} attempts")
                    self.successful_retries += 1
                
                return result
            
            except (ConnectionError, TimeoutError, OSError) as e:
                self.failed_attempts += 1
                
                if attempt < self.max_retries - 1:
                    base_wait = self.initial_backoff * (self.backoff_factor ** attempt)
                    # Add jitter: ±25% randomization to prevent thundering herd
                    wait_time = base_wait * (0.75 + random.random() * 0.5)
                    logger.warning(f"⏳ Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.exception(f"❌ All {self.max_retries} retry attempts failed")
            
            except Exception:
                # Non-network errors: fail immediately
                logger.exception("Non-retryable error")
                return None
        
        return None
    
    def get_stats(self) -> dict:
        """
        Get connection statistics.
        
        Returns:
            Dict with connection metrics
        """
        success_rate = ((self.total_calls - self.failed_attempts) / self.total_calls * 100) if self.total_calls > 0 else 100
        
        return {
            'failed_attempts': self.failed_attempts,
            'successful_retries': self.successful_retries,
            'total_calls': self.total_calls,
            'success_rate': f"{success_rate:.1f}%",
            'max_retries': self.max_retries
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"ConnectionHandler(failed={stats['failed_attempts']}, retried={stats['successful_retries']}, success_rate={stats['success_rate']})"


# Global connection handler instance
CONNECTION_HANDLER = ConnectionHandler(max_retries=3, initial_backoff=0.5, backoff_factor=2.0)
