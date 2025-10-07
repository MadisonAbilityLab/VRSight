"""
Object Detection Engine handling YOLO-based object detection with error recovery and performance monitoring.
"""

import time
import cv2
import torch
import threading
import queue
from ultralytics import YOLO

from settings import (
    YOLO_MODEL_PATH, DEVICE, MAX_CONSECUTIVE_ERRORS,
    MAX_INFERENCE_TIME, MAX_STAT_HISTORY
)


class ObjectDetectionEngine:
    """Handles object detection with YOLO model including error recovery and performance tracking."""

    def __init__(self, model_path=None, device=None):
        """Initialize the object detection engine."""
        self.model_path = model_path or YOLO_MODEL_PATH
        self.device = device or DEVICE
        self.model = None
        self.performance_stats = {
            'inference_time': [],
            'fps': [],
            'error_rate': 0
        }
        self.error_count = 0
        self.last_success_time = time.time()

    def initialize_model(self):
        """Initialize or reinitialize the YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            self.model.model.to(self.device)

            # Compile model for additional performance boost
            if hasattr(torch, 'compile') and self.device == 'cuda':
                try:
                    # Check if Triton is available for compilation
                    import triton
                    self.model.model = torch.compile(self.model.model, mode='max-autotune')
                    print("YOLO model compiled successfully")
                except ImportError:
                    print("Triton not available, skipping YOLO model compilation")
                except Exception as e:
                    print(f"YOLO model compilation failed: {e}")
                    pass  # Fall back to uncompiled if compilation fails

            # Simple GPU setup without complex pre-allocation
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                print("GPU memory cleared")

            print(f"YOLO model initialized on {self.device}")
            return True
        except Exception as e:
            print(f"Failed to initialize YOLO model: {e}")
            return False

    def process_frame(self, frame):
        """
        Process a single frame through YOLO detection.

        Args:
            frame: Input frame for detection

        Returns:
            tuple: (detections_list, annotated_frame, fps) or (None, None, 0) on error
        """
        if self.model is None:
            if not self.initialize_model():
                return None, None, 0

        # Validate frame
        if frame is None or frame.shape[0] <= 0 or frame.shape[1] <= 0 or len(frame.shape) != 3:
            return None, None, 0

        start_time = time.perf_counter()

        # Set up inference timeout monitoring
        inference_timer = threading.Timer(MAX_INFERENCE_TIME, lambda: None)
        inference_timer.start()

        try:
            # Run YOLO detection
            from config import get_config
            config = get_config()
            use_half = config.models.model_precision == 'fp16' and self.device == 'cuda'

            inference_start = time.perf_counter()
            results = self.model(frame, verbose=False, half=use_half)
            inference_time = time.perf_counter() - inference_start
            pure_inference_fps = 1 / inference_time if inference_time > 0 else 0

            detections = self._extract_detections(results, frame)

            inference_timer.cancel()
            self.last_success_time = time.time()
            self.error_count = 0

            annotated_frame = results[0].plot()

            total_time = time.perf_counter() - start_time
            pipeline_fps = 1 / total_time if total_time > 0 else 0

            self._update_performance_stats(inference_time, pure_inference_fps)

            avg_inference_fps = self._get_average_fps()
            cv2.putText(annotated_frame, f'Inference: {avg_inference_fps:.1f} FPS', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Pipeline: {pipeline_fps:.1f} FPS', (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            return detections, annotated_frame, pure_inference_fps

        except Exception as e:
            inference_timer.cancel()
            self.error_count += 1
            print(f"ERROR in object detection inference: {e}")

            # Check for stuck model
            time_since_success = time.time() - self.last_success_time
            if time_since_success > 5.0:
                print("Possible stuck model detected, attempting recovery...")
                self._attempt_model_recovery()

            return None, None, 0

    def _extract_detections(self, results, frame):
        """
        Extract detection data from YOLO results.

        Args:
            results: YOLO model output
            frame: Original input frame

        Returns:
            list: Detection dictionaries with bbox, confidence, class_id, label, pixels
        """
        detections = []

        try:
            for result in results:
                for detection in result.boxes:
                    try:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())

                        # Clamp coordinates to frame boundaries
                        x1 = max(0, min(x1, frame.shape[1] - 1))
                        y1 = max(0, min(y1, frame.shape[0] - 1))
                        x2 = max(0, min(x2, frame.shape[1] - 1))
                        y2 = max(0, min(y2, frame.shape[0] - 1))

                        # Skip invalid boxes
                        if x1 >= x2 or y1 >= y2:
                            continue

                        # Extract detection metadata
                        confidence = float(detection.conf)
                        class_id = int(detection.cls)
                        label = result.names[class_id]

                        # Extract object pixels
                        object_pixels = frame[y1:y2, x1:x2]

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'label': label,
                            'pixels': object_pixels
                        })

                    except Exception as e:
                        print(f"Error processing individual detection: {e}")
                        continue

        except Exception as e:
            print(f"Error extracting detections: {e}")

        return detections

    def _update_performance_stats(self, inference_time, fps):
        """Update performance statistics."""
        self.performance_stats['inference_time'].append(inference_time)
        self.performance_stats['fps'].append(fps)

        # Keep only recent values
        if len(self.performance_stats['inference_time']) > MAX_STAT_HISTORY:
            self.performance_stats['inference_time'] = self.performance_stats['inference_time'][-MAX_STAT_HISTORY:]
            self.performance_stats['fps'] = self.performance_stats['fps'][-MAX_STAT_HISTORY:]

        # Calculate error rate
        total_frames = len(self.performance_stats['fps']) + self.error_count
        if total_frames > 0:
            self.performance_stats['error_rate'] = self.error_count / total_frames

    def _get_average_fps(self, window=10):
        """Get average FPS over recent frames."""
        recent_fps = self.performance_stats['fps'][-window:]
        return sum(recent_fps) / len(recent_fps) if recent_fps else 0

    def _attempt_model_recovery(self):
        """Attempt to recover from model errors."""
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reinitialize model if error count is high
            if self.error_count >= 3:
                if self.initialize_model():
                    print("Model reinitialized during recovery")
                else:
                    print("Model reinitialization failed during recovery")

        except Exception as e:
            print(f"Error during model recovery: {e}")

    def should_reinitialize(self):
        """Check if model should be reinitialized due to errors."""
        return self.error_count >= MAX_CONSECUTIVE_ERRORS

    def get_performance_stats(self):
        """Get current performance statistics."""
        return self.performance_stats.copy()


def run_object_detection_thread(engine, frame_source, result_callback,
                               thread_coordinator, running_flag):
    """
    Run object detection in a dedicated thread.

    Args:
        engine: ObjectDetectionEngine instance
        frame_source: Function that returns the current frame
        result_callback: Function called with (detections, annotated_frame)
        thread_coordinator: ThreadCoordinator instance for monitoring
        running_flag: Shared flag indicating if thread should continue
    """
    thread_name = "object_detection"
    thread_coordinator.register_thread(thread_name)
    print("Object detection thread started.")

    while running_flag[0]:
        try:
            thread_coordinator.heartbeat(thread_name)

            frame = frame_source()

            if frame is None:
                time.sleep(0.1)
                continue

            process_start = time.perf_counter()

            detections, annotated_frame, fps = engine.process_frame(frame)

            if detections is not None and annotated_frame is not None:
                result_callback(detections, annotated_frame)

            if engine.should_reinitialize():
                print("Multiple errors in object detection, attempting recovery...")
                if engine.initialize_model():
                    engine.error_count = 0
                    thread_coordinator.record_restart(thread_name)
                    print("Model reinitialized successfully")
                else:
                    print("Failed to reinitialize model")
                    thread_coordinator.increment_error(thread_name)

            process_time = time.perf_counter() - process_start
            sleep_time = max(0.01, 0.05 - process_time if process_time < 0.05 else 0.01)
            time.sleep(sleep_time)

        except Exception as e:
            thread_coordinator.increment_error(thread_name)
            print(f"ERROR in object detection thread: {e}")
            time.sleep(1)