"""
Performance monitoring for inference service.
"""

import time
from typing import Dict, List, Any
from collections import deque
from datetime import datetime


class PerformanceMonitor:
    """
    Monitor inference performance metrics.
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> with monitor.track_prediction():
        ...     result = categorizer.predict("urgent meeting")
        >>> stats = monitor.get_statistics()
        >>> print(f"Average time: {stats['avg_time_ms']:.2f}ms")
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of recent predictions to track
        """
        self.window_size = window_size
        self.prediction_times = deque(maxlen=window_size)
        self.total_predictions = 0
        self.start_time = datetime.now()
    
    def track_prediction(self):
        """Context manager for tracking prediction time."""
        return PredictionTimer(self)
    
    def record_time(self, time_ms: float) -> None:
        """Record a prediction time."""
        self.prediction_times.append(time_ms)
        self.total_predictions += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.prediction_times:
            return {}
        
        times = list(self.prediction_times)
        times.sort()
        
        return {
            'total_predictions': self.total_predictions,
            'window_predictions': len(times),
            'avg_time_ms': sum(times) / len(times),
            'min_time_ms': times[0],
            'max_time_ms': times[-1],
            'p50_time_ms': times[len(times) // 2],
            'p95_time_ms': times[int(len(times) * 0.95)],
            'p99_time_ms': times[int(len(times) * 0.99)],
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }
    
    def print_statistics(self) -> None:
        """Print performance statistics."""
        stats = self.get_statistics()
        
        if not stats:
            print("No predictions tracked yet")
            return
        
        print("\n" + "=" * 60)
        print("PERFORMANCE STATISTICS")
        print("=" * 60)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Average time: {stats['avg_time_ms']:.2f}ms")
        print(f"Min time: {stats['min_time_ms']:.2f}ms")
        print(f"Max time: {stats['max_time_ms']:.2f}ms")
        print(f"P50 (median): {stats['p50_time_ms']:.2f}ms")
        print(f"P95: {stats['p95_time_ms']:.2f}ms")
        print(f"P99: {stats['p99_time_ms']:.2f}ms")
        print(f"Uptime: {stats['uptime_seconds']:.0f}s")
        print("=" * 60 + "\n")


class PredictionTimer:
    """Context manager for timing predictions."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.monitor.record_time(elapsed_ms)
