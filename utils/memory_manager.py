"""
Layer 2: Memory Management System

Real-time memory monitoring with automatic cleanup.
Prevents memory overflow by tracking usage and triggering garbage collection.

Features:
- Memory usage monitoring (psutil)
- Auto-cleanup when threshold reached
- Cache expiration on demand
- Graceful error handling
"""

import gc
import psutil
from logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Monitor and manage memory usage on HF free tier (300MB limit).
    Automatically triggers cleanup when approaching limits.
    """
    
    def __init__(self, max_memory_mb: int = 280, cleanup_threshold_pct: float = 0.8):
        """
        Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum allowed memory in MB (HF limit ~300MB)
            cleanup_threshold_pct: Trigger cleanup at X% of max_memory
        """
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold_mb = max_memory_mb * cleanup_threshold_pct
        self.alerts_sent = 0
        logger.info(f"🚗 Memory manager initialized: Max={max_memory_mb}MB, Threshold={self.cleanup_threshold_mb:.0f}MB")
    
    def get_memory_usage(self) -> float:
        """
        Get current process memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return 0.0
    
    def check_and_cleanup(self) -> bool:
        """
        Check memory usage and trigger cleanup if needed.
        
        Returns:
            True if memory OK, False if critical
        """
        from utils.cache_manager import COMPUTATION_CACHE
        
        current_mem = self.get_memory_usage()
        
        # Check if approaching threshold
        if current_mem > self.cleanup_threshold_mb:
            logger.warning(f"\ud83d\udea8 Memory usage high ({current_mem:.0f}MB / {self.cleanup_threshold_mb:.0f}MB threshold)")
            
            # Clear expired cache entries
            expired_count = COMPUTATION_CACHE.clear_expired()
            
            # Force garbage collection
            gc.collect()
            
            new_mem = self.get_memory_usage()
            freed = current_mem - new_mem
            
            logger.info(f"\ud83d\udd02 Memory after cleanup: {new_mem:.0f}MB (freed {freed:.0f}MB, {expired_count} cache items removed)")
            
            # Check if still critical
            if new_mem > self.max_memory_mb:
                logger.error(f"\ud83d\udca3 CRITICAL: Memory {new_mem:.0f}MB > {self.max_memory_mb}MB limit!")
                self.alerts_sent += 1
                return False
        
        return True
    
    def get_memory_status(self) -> dict:
        """
        Get detailed memory status.
        
        Returns:
            Dict with memory metrics
        """
        current_mem = self.get_memory_usage()
        usage_pct = (current_mem / self.max_memory_mb * 100)
        
        return {
            'current_mb': f"{current_mem:.0f}",
            'max_mb': self.max_memory_mb,
            'usage_pct': f"{usage_pct:.1f}%",
            'threshold_mb': f"{self.cleanup_threshold_mb:.0f}",
            'status': 'OK' if usage_pct < 80 else 'WARNING' if usage_pct < 100 else 'CRITICAL'
        }
    
    def __repr__(self) -> str:
        status = self.get_memory_status()
        return f"MemoryManager({status['current_mb']}/{status['max_mb']}MB, {status['usage_pct']}%)" 


# Global memory manager instance
MEMORY_MANAGER = MemoryManager(max_memory_mb=280, cleanup_threshold_pct=0.8)
