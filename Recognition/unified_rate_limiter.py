"""
Unified Rate Limiting System to consolidate AdaptiveRateLimiter (GPT), TTSRateLimiter (speech), and CategoryCooldownManager.
"""

import time
import threading
import queue
from typing import Dict, Optional, Callable, List, Any
from enum import Enum


class LimiterType(Enum):
    """Types of rate limiting strategies."""
    ADAPTIVE = "adaptive"  # For API calls with backoff
    SEMAPHORE = "semaphore"  # For concurrent resource limiting
    COOLDOWN = "cooldown"  # For category-based cooldowns
    QUEUE = "queue"  # For queued processing


class RateLimitStrategy:
    """Base class for rate limiting strategies."""

    def __init__(self, name: str):
        self.name = name
        self.lock = threading.Lock()

    def can_proceed(self, identifier: str = None) -> bool:
        """Check if operation can proceed."""
        raise NotImplementedError

    def record_success(self, identifier: str = None):
        """Record successful operation."""
        pass

    def record_failure(self, identifier: str = None):
        """Record failed operation."""
        pass

    def wait_if_needed(self, identifier: str = None) -> bool:
        """Wait if rate limiting requires it."""
        return self.can_proceed(identifier)


class AdaptiveStrategy(RateLimitStrategy):
    """Adaptive rate limiting with exponential backoff for API calls."""

    def __init__(self, name: str, base_interval: float = 3.0, max_interval: float = 30.0):
        super().__init__(name)
        self.last_request_time = 0
        self.base_interval = base_interval
        self.current_interval = base_interval
        self.max_interval = max_interval
        self.success_count = 0
        self.failure_count = 0

    def wait_if_needed(self, identifier: str = None) -> bool:
        """Wait if needed based on adaptive rate limiting."""
        # Special bypass for certain identifiers
        if identifier == "menu":
            return True

        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.current_interval:
                wait_time = self.current_interval - time_since_last

            self.last_request_time = time.time()
            return True

    def record_success(self, identifier: str = None):
        """Record successful operation and decrease interval."""
        with self.lock:
            self.success_count += 1
            self.failure_count = 0

            # Gradually decrease interval after consecutive successes
            if self.success_count > 3 and self.current_interval > self.base_interval:
                self.current_interval = max(self.base_interval,
                                          self.current_interval * 0.8)

    def record_failure(self, identifier: str = None):
        """Record failed operation and increase interval."""
        with self.lock:
            self.failure_count += 1
            self.success_count = 0

            # Exponentially increase interval on failures
            if self.failure_count > 2:
                self.current_interval = min(self.max_interval,
                                          self.current_interval * 1.5)


class SemaphoreStrategy(RateLimitStrategy):
    """Semaphore-based limiting for concurrent resource access."""

    def __init__(self, name: str, max_concurrent: int = 2, cooldown_time: float = 0.5):
        super().__init__(name)
        self.semaphore = threading.Semaphore(max_concurrent)
        self.cooldown_time = cooldown_time
        self.active_operations = set()

    def can_proceed(self, identifier: str = None) -> bool:
        """Check if operation can proceed (acquire semaphore)."""
        acquired = self.semaphore.acquire(blocking=False)
        if acquired and identifier:
            with self.lock:
                self.active_operations.add(identifier)
        return acquired

    def record_success(self, identifier: str = None):
        """Release semaphore and apply cooldown."""
        self._release_semaphore(identifier)

    def record_failure(self, identifier: str = None):
        """Release semaphore on failure."""
        self._release_semaphore(identifier)

    def _release_semaphore(self, identifier: str = None):
        """Release the semaphore and clean up."""
        try:
            if identifier:
                with self.lock:
                    self.active_operations.discard(identifier)
            self.semaphore.release()
            # Small cooldown to avoid overwhelming the service
            time.sleep(self.cooldown_time)
        except Exception as e:
            print(f"Error releasing semaphore for {self.name}: {e}")


class CooldownStrategy(RateLimitStrategy):
    """Category-based cooldown limiting."""

    def __init__(self, name: str, cooldown_times: Dict[str, float] = None):
        super().__init__(name)
        self.cooldown_times = cooldown_times or {"default": 5.0}
        self.last_execution = {}
        self.text_history = {}

    def can_proceed(self, identifier: str = None) -> bool:
        """Check if enough time has passed since last execution."""
        if not identifier:
            return True

        with self.lock:
            current_time = time.time()

            # Get cooldown time for this category
            cooldown_time = self.cooldown_times.get(identifier,
                           self.cooldown_times.get("default", 5.0))

            last_time = self.last_execution.get(identifier, 0)
            time_since_last = current_time - last_time

            return time_since_last >= cooldown_time

    def record_success(self, identifier: str = None):
        """Record successful execution timestamp."""
        if identifier:
            with self.lock:
                self.last_execution[identifier] = time.time()

    def record_failure(self, identifier: str = None):
        """Failure doesn't affect cooldown timing."""
        pass


class QueueStrategy(RateLimitStrategy):
    """Queue-based processing with priority support."""

    def __init__(self, name: str, max_queue_size: int = 50):
        super().__init__(name)
        self.queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.worker_active = False
        self.worker_thread = None
        self.processor_callback = None

    def set_processor(self, callback: Callable):
        """Set the callback function to process queued items."""
        self.processor_callback = callback

    def can_proceed(self, identifier: str = None) -> bool:
        """Always allows queueing (up to queue limit)."""
        return not self.queue.full()

    def enqueue_item(self, item: Any, priority: int = 0, identifier: str = None):
        """Add item to processing queue."""
        try:
            self.queue.put((priority, time.time(), identifier, item), block=False)
            self._ensure_worker_running()
            return True
        except queue.Full:
            print(f"Queue full for {self.name}, dropping item")
            return False

    def _ensure_worker_running(self):
        """Ensure background worker thread is running."""
        with self.lock:
            if not self.worker_active and self.processor_callback:
                self.worker_active = True
                self.worker_thread = threading.Thread(
                    target=self._worker_loop, daemon=True
                )
                self.worker_thread.start()

    def _worker_loop(self):
        """Background worker to process queued items."""
        while self.worker_active or not self.queue.empty():
            try:
                # Wait for items with timeout
                priority, timestamp, identifier, item = self.queue.get(timeout=1.0)

                if self.processor_callback:
                    try:
                        self.processor_callback(item, identifier)
                        self.record_success(identifier)
                    except Exception as e:
                        print(f"Error processing queued item: {e}")
                        self.record_failure(identifier)

                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in queue worker: {e}")

        with self.lock:
            self.worker_active = False

    def clear_queue(self):
        """Clear all pending items from queue."""
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        except queue.Empty:
            pass

    def stop_worker(self):
        """Stop the background worker thread."""
        self.worker_active = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)


class UnifiedRateLimiter:
    """
    Unified rate limiting system that combines different limiting strategies.
    """

    def __init__(self):
        self.strategies: Dict[str, RateLimitStrategy] = {}
        self.default_strategy = None
        self.lock = threading.Lock()

        # Initialize default strategies
        self._setup_default_strategies()

    def _setup_default_strategies(self):
        """Set up commonly used rate limiting strategies."""
        # GPT API rate limiting (adaptive)
        self.add_strategy("gpt_api", AdaptiveStrategy("gpt_api", base_interval=3.0, max_interval=30.0))

        # TTS service limiting (semaphore)
        self.add_strategy("tts", SemaphoreStrategy("tts", max_concurrent=2, cooldown_time=0.5))

        # Speech cooldowns (cooldown-based)
        cooldown_times = {
            "special_labels": 5.0,
            "user_command": 1.0,
            "guardian": 5.0,
            "out of bounds": 5.0,
            "progress bar": 15.0,
            "dashboard": 5.0,
            "default": 5.0
        }
        self.add_strategy("speech", CooldownStrategy("speech", cooldown_times))

        # Speech queue processing
        self.add_strategy("speech_queue", QueueStrategy("speech_queue", max_queue_size=50))

    def add_strategy(self, name: str, strategy: RateLimitStrategy):
        """Add a new rate limiting strategy."""
        with self.lock:
            self.strategies[name] = strategy
            if self.default_strategy is None:
                self.default_strategy = name

    def get_strategy(self, name: str) -> Optional[RateLimitStrategy]:
        """Get a specific rate limiting strategy."""
        return self.strategies.get(name)

    def can_proceed(self, strategy_name: str, identifier: str = None) -> bool:
        """Check if operation can proceed with specified strategy."""
        strategy = self.get_strategy(strategy_name)
        if strategy:
            return strategy.can_proceed(identifier)
        return True

    def wait_if_needed(self, strategy_name: str, identifier: str = None) -> bool:
        """Wait if rate limiting requires it."""
        strategy = self.get_strategy(strategy_name)
        if strategy:
            return strategy.wait_if_needed(identifier)
        return True

    def record_success(self, strategy_name: str, identifier: str = None):
        """Record successful operation."""
        strategy = self.get_strategy(strategy_name)
        if strategy:
            strategy.record_success(identifier)

    def record_failure(self, strategy_name: str, identifier: str = None):
        """Record failed operation."""
        strategy = self.get_strategy(strategy_name)
        if strategy:
            strategy.record_failure(identifier)

    def clear_queue(self, strategy_name: str):
        """Clear queue for queue-based strategies."""
        strategy = self.get_strategy(strategy_name)
        if isinstance(strategy, QueueStrategy):
            strategy.clear_queue()

    def enqueue_item(self, strategy_name: str, item: Any, priority: int = 0, identifier: str = None) -> bool:
        """Enqueue item for queue-based strategies."""
        strategy = self.get_strategy(strategy_name)
        if isinstance(strategy, QueueStrategy):
            return strategy.enqueue_item(item, priority, identifier)
        return False

    def set_queue_processor(self, strategy_name: str, callback: Callable):
        """Set processor callback for queue-based strategies."""
        strategy = self.get_strategy(strategy_name)
        if isinstance(strategy, QueueStrategy):
            strategy.set_processor(callback)

    def shutdown(self):
        """Shutdown all strategies and cleanup resources."""
        with self.lock:
            for strategy in self.strategies.values():
                if isinstance(strategy, QueueStrategy):
                    strategy.stop_worker()


# Global instance for easy access
unified_limiter = UnifiedRateLimiter()


# Convenience functions for backward compatibility
def get_gpt_limiter() -> AdaptiveStrategy:
    """Get the GPT API rate limiter."""
    return unified_limiter.get_strategy("gpt_api")


def get_tts_limiter() -> SemaphoreStrategy:
    """Get the TTS service rate limiter."""
    return unified_limiter.get_strategy("tts")


def get_speech_cooldown() -> CooldownStrategy:
    """Get the speech cooldown manager."""
    return unified_limiter.get_strategy("speech")


def get_speech_queue() -> QueueStrategy:
    """Get the speech queue processor."""
    return unified_limiter.get_strategy("speech_queue")