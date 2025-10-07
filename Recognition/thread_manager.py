"""
Thread Management consolidating thread coordination, monitoring, and resource management.
"""

import threading
import time
import queue
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any

from settings import (
    THREAD_HEARTBEAT_TIMEOUT, THREAD_ERROR_THRESHOLD,
    MAX_THREAD_RESTARTS, THREAD_SHUTDOWN_TIMEOUT
)


@dataclass
class ThreadStatus:
    """Thread status information"""
    last_heartbeat: float
    error_count: int
    restart_attempts: int
    is_critical: bool = True


class ThreadSafeResourceManager:
    """Manages shared resources with per-resource locking for minimal contention"""

    def __init__(self):
        self._locks = {
            'frame': threading.Lock(),
            'object_detections': threading.Lock(),
            'edge_detections': threading.Lock(),
            'depth_map': threading.Lock()
        }
        self._resources = {
            'frame': None,
            'object_detections': None,
            'edge_detections': None,
            'depth_map': None
        }

    @contextmanager
    def resource_access(self, *resource_names):
        locks = [self._locks[name] for name in resource_names if name in self._locks]
        for lock in locks:
            lock.acquire()
        try:
            yield {name: self._resources.get(name) for name in resource_names}
        finally:
            for lock in locks:
                lock.release()

    def update_resource(self, name: str, value: Any):
        if name in self._locks:
            with self._locks[name]:
                self._resources[name] = value

    def get_resource(self, name: str) -> Any:
        if name in self._locks:
            with self._locks[name]:
                return self._resources.get(name)
        return None

    def copy_resource(self, name: str) -> Any:
        if name in self._locks:
            with self._locks[name]:
                resource = self._resources.get(name)
                return resource.copy() if hasattr(resource, 'copy') and resource is not None else resource
        return None


class ThreadManager:
    """Enhanced thread coordination with unified resource management"""

    def __init__(self):
        self._lock = threading.RLock()
        self._thread_status: Dict[str, ThreadStatus] = {}
        self._shutdown_flag = False
        self._running_threads = {}
        self._resource_manager = ThreadSafeResourceManager()

        # Queue management
        self._queues = {
            'object_detection': queue.Queue(maxsize=10),
            'depth_detection': queue.Queue(maxsize=10),
            'edge_detection': queue.Queue(maxsize=10),
            'gpt_request': queue.Queue()
        }

    @property
    def resource_manager(self) -> ThreadSafeResourceManager:
        """Access to the resource manager"""
        return self._resource_manager

    def register_thread(self, name: str, is_critical: bool = True):
        """Register a thread for monitoring"""
        with self._lock:
            self._thread_status[name] = ThreadStatus(
                last_heartbeat=time.time(),
                error_count=0,
                restart_attempts=0,
                is_critical=is_critical
            )

    def heartbeat(self, name: str):
        """Update thread heartbeat"""
        with self._lock:
            if name in self._thread_status:
                self._thread_status[name].last_heartbeat = time.time()

    def increment_error(self, name: str):
        """Record an error for a thread"""
        with self._lock:
            if name in self._thread_status:
                self._thread_status[name].error_count += 1

    def needs_restart(self, name: str) -> bool:
        """Check if thread needs restart"""
        with self._lock:
            if name not in self._thread_status:
                return False

            status = self._thread_status[name]
            current_time = time.time()
            time_since_heartbeat = current_time - status.last_heartbeat

            return (time_since_heartbeat > THREAD_HEARTBEAT_TIMEOUT or
                    status.error_count >= THREAD_ERROR_THRESHOLD)

    def record_restart(self, name: str) -> bool:
        """Record a thread restart attempt. Returns False if max restarts reached."""
        with self._lock:
            if name not in self._thread_status:
                return False

            status = self._thread_status[name]

            if status.restart_attempts >= MAX_THREAD_RESTARTS:
                print(f"WARNING: Thread '{name}' has exceeded maximum restarts")
                if status.is_critical:
                    self._shutdown_flag = True
                    print("Setting shutdown flag due to critical thread failure")
                return False
            else:
                status.restart_attempts += 1
                status.error_count = 0
                status.last_heartbeat = time.time()
                print(f"Thread '{name}' restarted (attempt {status.restart_attempts})")
                return True

    def should_shutdown(self) -> bool:
        """Check if system should shut down"""
        return self._shutdown_flag

    def safe_shutdown(self):
        """Signal all threads to shutdown safely"""
        with self._lock:
            self._shutdown_flag = True
            print("Thread manager signaling safe shutdown")

    def start_thread(self, target: Callable, name: str, daemon: bool = True,
                     is_critical: bool = True, args: tuple = ()) -> threading.Thread:
        """Start and register a thread"""
        self.register_thread(name, is_critical)

        thread = threading.Thread(target=target, name=name, daemon=daemon, args=args)
        thread.start()

        with self._lock:
            self._running_threads[name] = thread

        return thread

    def get_queue(self, queue_name: str) -> queue.Queue:
        """Get a managed queue"""
        return self._queues.get(queue_name)

    def clear_queues(self):
        """Clear all managed queues"""
        for queue_name, q in self._queues.items():
            try:
                while not q.empty():
                    q.get_nowait()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error clearing queue {queue_name}: {e}")

    def detect_deadlocks(self) -> List[str]:
        """Detect potential deadlocks in critical threads"""
        with self._lock:
            current_time = time.time()
            deadlocked = []

            critical_threads = ["object_detection", "depth_detection", "edge_detection"]

            for name in critical_threads:
                if name in self._thread_status:
                    status = self._thread_status[name]
                    time_since_heartbeat = current_time - status.last_heartbeat

                    if time_since_heartbeat > THREAD_HEARTBEAT_TIMEOUT:
                        deadlocked.append(name)

            return deadlocked

    def emergency_restart(self, deadlocked_threads: List[str]):
        """Emergency restart for deadlocked threads"""
        print(f"Emergency restart for threads: {deadlocked_threads}")

        # Clear queues to unblock threads
        self.clear_queues()

        # Force restart by setting old heartbeat and high error count
        with self._lock:
            for thread_name in deadlocked_threads:
                if thread_name in self._thread_status:
                    status = self._thread_status[thread_name]
                    status.last_heartbeat = 0
                    status.error_count = THREAD_ERROR_THRESHOLD

        print("Emergency restart preparations complete")

    def get_status_report(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive status report"""
        with self._lock:
            report = {}
            for name, status in self._thread_status.items():
                report[name] = {
                    'last_heartbeat': status.last_heartbeat,
                    'error_count': status.error_count,
                    'restart_attempts': status.restart_attempts,
                    'is_critical': status.is_critical,
                    'time_since_heartbeat': time.time() - status.last_heartbeat
                }
            return report

    def cleanup(self):
        """Cleanup resources and stop threads"""
        self.safe_shutdown()

        # Wait for threads to finish
        with self._lock:
            for name, thread in self._running_threads.items():
                if thread.is_alive():
                    thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)
                    if thread.is_alive():
                        print(f"Warning: Thread {name} did not shutdown cleanly")

        self.clear_queues()
        print("Thread manager cleanup complete")


# Global instance - singleton pattern for system-wide coordination
_thread_manager_instance = None

def get_thread_manager() -> ThreadManager:
    """Get the global thread manager instance"""
    global _thread_manager_instance
    if _thread_manager_instance is None:
        _thread_manager_instance = ThreadManager()
    return _thread_manager_instance