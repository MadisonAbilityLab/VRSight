import time
import psutil
from simple_logger import log_info

class SimplePerformanceMonitor:
    """Basic performance monitoring without over-engineering."""

    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_check = time.time()
        self.last_memory_check = time.time()

    def update_frame_count(self):
        """Update frame count for FPS calculation."""
        self.frame_count += 1

    def get_fps(self):
        """Get current FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_fps_check
        if elapsed >= 1.0:  # Update every second
            fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_check = current_time
            return fps
        return None

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def report_performance(self, force=False):
        """Report performance metrics (called periodically)."""
        current_time = time.time()

        # Report every 60 seconds or when forced
        if force or (current_time - self.last_memory_check >= 60):
            fps = self.get_fps()
            memory_mb = self.get_memory_usage()
            uptime = current_time - self.start_time

            if fps is not None:
                log_info(f"Performance: {fps:.1f} FPS, {memory_mb:.1f}MB RAM, {uptime:.0f}s uptime")
            else:
                log_info(f"Performance: {memory_mb:.1f}MB RAM, {uptime:.0f}s uptime")

            self.last_memory_check = current_time

# Global performance monitor
performance_monitor = SimplePerformanceMonitor()

def update_frame_count():
    """Update frame count (call this for each processed frame)."""
    performance_monitor.update_frame_count()

def report_performance(force=False):
    """Report current performance metrics."""
    performance_monitor.report_performance(force)

def get_memory_usage():
    """Get current memory usage in MB."""
    return performance_monitor.get_memory_usage()