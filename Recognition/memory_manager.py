"""
Memory Management consolidating memory monitoring, cleanup, and optimization across engines.
"""

import gc
import time
import threading
import psutil
import torch
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque

from settings import MEMORY_CLEANUP_THRESHOLD_MB, MEMORY_CLEANUP_INTERVAL


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_mb: float
    used_mb: float
    available_mb: float
    percent_used: float
    gpu_mb: Optional[float] = None
    gpu_cached_mb: Optional[float] = None


class FrameBuffer:
    """Optimized frame buffering with memory management"""

    def __init__(self, max_size: int = 5):
        self._buffer = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._max_size = max_size

    def add_frame(self, frame):
        """Add frame to buffer (thread-safe)"""
        with self._lock:
            self._buffer.append(frame)

    def get_latest_frame(self):
        """Get the most recent frame"""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_frame_copy(self):
        """Get a copy of the latest frame"""
        with self._lock:
            frame = self._buffer[-1] if self._buffer else None
            return frame.copy() if frame is not None else None

    def clear(self):
        """Clear the frame buffer"""
        with self._lock:
            self._buffer.clear()

    def size(self) -> int:
        """Get current buffer size"""
        with self._lock:
            return len(self._buffer)


class MemoryManager:
    """Advanced memory management with predictive cleanup and leak detection"""

    def __init__(self, cleanup_threshold_mb: int = MEMORY_CLEANUP_THRESHOLD_MB):
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.cleanup_interval = MEMORY_CLEANUP_INTERVAL
        self.last_cleanup_time = time.time()
        self._lock = threading.RLock()

        # Memory tracking
        self._memory_history = deque(maxlen=100)  # Last 100 memory readings
        self._cleanup_callbacks: List[Callable] = []

        # Frame buffering
        self._frame_buffers: Dict[str, FrameBuffer] = {
            'raw_frames': FrameBuffer(max_size=3),
            'object_frames': FrameBuffer(max_size=2),
            'depth_frames': FrameBuffer(max_size=2),
            'edge_frames': FrameBuffer(max_size=2)
        }

    def register_cleanup_callback(self, callback: Callable):
        """Register a callback for memory cleanup events"""
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()

        gpu_mb = None
        gpu_cached_mb = None

        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                gpu_mb = gpu_memory.get('allocated_bytes.all.current', 0) / (1024 * 1024)
                gpu_cached_mb = gpu_memory.get('reserved_bytes.all.current', 0) / (1024 * 1024)
            except Exception:
                pass

        stats = MemoryStats(
            total_mb=memory.total / (1024 * 1024),
            used_mb=memory.used / (1024 * 1024),
            available_mb=memory.available / (1024 * 1024),
            percent_used=memory.percent,
            gpu_mb=gpu_mb,
            gpu_cached_mb=gpu_cached_mb
        )

        # Track memory history
        with self._lock:
            self._memory_history.append((time.time(), stats.used_mb))

        return stats

    def detect_memory_leak(self, window_minutes: int = 5) -> bool:
        """Detect if there's a memory leak based on trend analysis"""
        with self._lock:
            if len(self._memory_history) < 10:  # Need sufficient data
                return False

            # Get memory readings within the time window
            current_time = time.time()
            window_start = current_time - (window_minutes * 60)

            recent_readings = [
                (timestamp, memory_mb) for timestamp, memory_mb in self._memory_history
                if timestamp >= window_start
            ]

            if len(recent_readings) < 5:
                return False

            # Check for consistent upward trend
            memory_values = [memory_mb for _, memory_mb in recent_readings]

            # Simple trend detection: check if memory increased significantly
            start_memory = memory_values[0]
            end_memory = memory_values[-1]
            increase = end_memory - start_memory

            # Consider it a leak if memory increased by more than 200MB in the window
            return increase > 200

    def cleanup_gpu_memory(self) -> bool:
        """Clean up GPU memory if available"""
        if not torch.cuda.is_available():
            return False

        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
        except Exception as e:
            print(f"Error cleaning GPU memory: {e}")
            return False

    def cleanup_system_memory(self) -> bool:
        """Clean up system memory"""
        try:
            # Force garbage collection
            collected = gc.collect()

            # Clear frame buffers if memory pressure is high
            stats = self.get_memory_stats()
            if stats.percent_used > 85:  # High memory usage
                for buffer in self._frame_buffers.values():
                    buffer.clear()
                print("Cleared frame buffers due to high memory usage")

            return True
        except Exception as e:
            print(f"Error in system memory cleanup: {e}")
            return False

    def check_and_cleanup(self, force: bool = False) -> bool:
        """Check memory usage and cleanup if needed"""
        current_time = time.time()

        # Check if cleanup is needed
        if not force and (current_time - self.last_cleanup_time < self.cleanup_interval):
            return False

        stats = self.get_memory_stats()

        # Determine if cleanup is necessary
        needs_cleanup = (
            force or
            stats.used_mb > self.cleanup_threshold_mb or
            stats.percent_used > 80 or
            self.detect_memory_leak()
        )

        if not needs_cleanup:
            return False

        print(f"Memory cleanup triggered - Used: {stats.used_mb:.1f}MB ({stats.percent_used:.1f}%)")

        # Perform cleanup
        gpu_cleaned = self.cleanup_gpu_memory()
        system_cleaned = self.cleanup_system_memory()

        # Call registered cleanup callbacks
        with self._lock:
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in cleanup callback: {e}")

        self.last_cleanup_time = current_time

        # Get post-cleanup stats
        post_stats = self.get_memory_stats()
        freed_mb = stats.used_mb - post_stats.used_mb

        print(f"Memory cleanup complete - Freed: {freed_mb:.1f}MB, "
              f"New usage: {post_stats.used_mb:.1f}MB ({post_stats.percent_used:.1f}%)")

        return True

    def get_frame_buffer(self, buffer_name: str) -> Optional[FrameBuffer]:
        """Get a frame buffer by name"""
        return self._frame_buffers.get(buffer_name)

    def add_frame_to_buffer(self, buffer_name: str, frame):
        """Add frame to specified buffer"""
        buffer = self._frame_buffers.get(buffer_name)
        if buffer:
            buffer.add_frame(frame)

    def get_memory_report(self) -> Dict:
        """Get comprehensive memory report"""
        stats = self.get_memory_stats()

        # Calculate memory trend
        trend = "stable"
        with self._lock:
            if len(self._memory_history) >= 2:
                recent_memory = self._memory_history[-1][1]
                old_memory = self._memory_history[max(0, len(self._memory_history) - 10)][1]
                if recent_memory > old_memory + 50:
                    trend = "increasing"
                elif recent_memory < old_memory - 50:
                    trend = "decreasing"

        # Frame buffer stats
        buffer_stats = {}
        for name, buffer in self._frame_buffers.items():
            buffer_stats[name] = {
                'size': buffer.size(),
                'max_size': buffer._max_size
            }

        return {
            'memory_stats': stats,
            'trend': trend,
            'leak_detected': self.detect_memory_leak(),
            'buffer_stats': buffer_stats,
            'last_cleanup': time.time() - self.last_cleanup_time
        }

    def optimize_for_performance(self):
        """Apply performance optimizations"""
        # Set aggressive garbage collection
        gc.set_threshold(700, 10, 10)  # More frequent GC

        # GPU optimizations
        if torch.cuda.is_available():
            try:
                # Enable memory fraction to prevent over-allocation
                torch.cuda.set_per_process_memory_fraction(0.9)
                # Enable memory mapping
                torch.cuda.memory._set_allocator_settings('expandable_segments:True')
            except Exception as e:
                print(f"GPU optimization warning: {e}")

        print("Memory performance optimizations applied")


# Global instance
_memory_manager_instance = None

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance