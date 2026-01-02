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
        Create a ConnectionHandler configured for exponential-backoff retry behavior.
        
        Parameters:
            max_retries (int): Total number of attempts (initial call + retries) to make before giving up.
            initial_backoff (float): Base wait time in seconds used before the first retry.
            backoff_factor (float): Multiplier applied to the backoff between successive retries.
        
        Initializes internal counters:
            failed_attempts, successful_retries, total_calls
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
        Execute a callable and retry on network-related failures using exponential backoff with jitter.
        
        Retries on ConnectionError, TimeoutError, and OSError up to `max_retries`. Increments `total_calls` on entry, increments `failed_attempts` for each retryable failure, and increments `successful_retries` when a call succeeds after one or more retries. Non-retryable exceptions cause an immediate return of `None`.
        
        Parameters:
            func (Callable): The callable to invoke.
            *args: Positional arguments forwarded to `func`.
            **kwargs: Keyword arguments forwarded to `func`.
        
        Returns:
            The value returned by `func` if it succeeds; `None` if all retry attempts fail or a non-retryable error occurs.
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
        Provide connection retry and call statistics.
        
        Returns:
            dict: Mapping of statistics:
                - 'failed_attempts': number of failed retry attempts.
                - 'successful_retries': number of calls that succeeded after one or more retries.
                - 'total_calls': total number of calls attempted through the handler.
                - 'success_rate': success rate as a percentage string with one decimal (e.g., "92.3%"); when `total_calls` is 0 this is "100.0%".
                - 'max_retries': configured maximum retry attempts.
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
        """
        Return a short string summarizing the handler's current retry statistics.
        
        Returns:
            str: A representation in the form "ConnectionHandler(failed=<failed>, retried=<retried>, success_rate=<percent>)".
        """
        stats = self.get_stats()
        return f"ConnectionHandler(failed={stats['failed_attempts']}, retried={stats['successful_retries']}, success_rate={stats['success_rate']})"


# Global connection handler instance
CONNECTION_HANDLER = ConnectionHandler(max_retries=3, initial_backoff=0.5, backoff_factor=2.0)