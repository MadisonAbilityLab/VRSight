"""
Depth Detection Engine for VR-AI Scene Recognition.
Handles DepthAnything V2 model for depth estimation with error recovery.
"""

import time
import cv2
import torch
import torch._dynamo
import torch.nn.functional as F
import numpy as np
import threading
from torchvision.transforms import Compose

# Suppress torch._dynamo errors and fall back to eager execution
torch._dynamo.config.suppress_errors = True

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

from settings import (
    CAM_WIDTH, CAM_HEIGHT, DEVICE, MAX_CONSECUTIVE_ERRORS
)


class DepthDetectionEngine:
    """Handles depth estimation using DepthAnything V2 model."""

    def __init__(self, encoder='vits', device=None):
        """Initialize the depth detection engine."""
        self.encoder = encoder
        self.device = device or DEVICE
        self.model = None
        self.transform = None
        self.error_count = 0
        self.fps_values = []
        self.fps_calc_start_time = time.time()
        self.use_fp16 = False  # Initialize flag

        # Depth model configurations
        self.depth_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self._initialize_transform()

    def _initialize_transform(self):
        """Initialize the image preprocessing transform."""
        self.transform = Compose([
            Resize(
                width=CAM_WIDTH,
                height=CAM_HEIGHT,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def initialize_model(self):
        """Initialize or reinitialize the depth model."""
        try:
            print(f"Initializing DepthAnything V2 model with encoder: {self.encoder}")

            # Create model with configuration
            config = self.depth_configs[self.encoder]
            print("Creating DepthAnything V2 model...")
            self.model = DepthAnythingV2(**config)

            # Load pretrained weights
            checkpoint_path = f'checkpoints/depth_anything_v2_{self.encoder}.pth'
            print(f"Loading weights from {checkpoint_path}...")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

            # Move to device and set to evaluation mode
            print(f"Moving model to {self.device}...")
            self.model = self.model.to(self.device).eval()
            print("Model moved to device and set to eval mode")

            # Apply precision optimization from config
            from config import get_config
            config = get_config()
            self.use_fp16 = config.models.model_precision == 'fp16' and self.device == 'cuda'

            # Check if torch.compile is available and working
            can_compile = False
            if hasattr(torch, 'compile') and self.device == 'cuda':
                try:
                    # Check if Triton is available
                    import triton
                    can_compile = True
                except ImportError:
                    print("Triton not available, skipping model compilation")
                    can_compile = False

            # Compile model first, then convert to FP16 to avoid dtype issues
            compiled_successfully = False
            if can_compile:
                try:
                    self.model = torch.compile(self.model, mode='max-autotune')
                    compiled_successfully = True
                    print("Depth model compiled successfully")
                except Exception as e:
                    print(f"Depth model compilation failed: {e}")
                    pass  # Fall back to uncompiled

            # Convert to FP16 after compilation to maintain dtype consistency
            if self.use_fp16:
                try:
                    self.model = self.model.half()  # Convert to FP16 for 2x speed boost
                    print("Depth model converted to FP16")
                except Exception as e:
                    print(f"FP16 conversion failed, using FP32: {e}")
                    self.use_fp16 = False

            print(f"Depth model initialized successfully on {self.device}")
            return True

        except Exception as e:
            print(f"Failed to initialize depth model: {e}")
            return False

    def process_frame(self, frame):
        """
        Process a single frame through depth estimation.

        Args:
            frame: Input BGR frame

        Returns:
            tuple: (depth_map, depth_visualization, fps) or (None, None, 0) on error
        """
        if self.model is None:
            if not self.initialize_model():
                return None, None, 0

        # Validate frame
        if frame is None or frame.shape[0] <= 0 or frame.shape[1] <= 0 or len(frame.shape) != 3:
            return None, None, 0

        start_time = time.perf_counter()

        try:
            # Preprocess frame for depth model
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frame_height, frame_width = processed_frame.shape[:2]

            # Apply transforms
            processed_frame = self.transform({'image': processed_frame})['image']
            processed_frame = torch.from_numpy(processed_frame).unsqueeze(0).to(self.device)
            if self.use_fp16:
                processed_frame = processed_frame.half()

            inference_start = time.perf_counter()
            with torch.inference_mode():
                depth = self.model(processed_frame)
            inference_time = time.perf_counter() - inference_start
            pure_inference_fps = 1 / inference_time if inference_time > 0 else 0

            depth = F.interpolate(
                depth[None],
                (frame_height, frame_width),
                mode='bilinear',
                align_corners=False
            )[0, 0]

            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_map = depth_normalized.to(torch.uint8).cpu().numpy()

            depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

            total_time = time.perf_counter() - start_time
            pipeline_fps = 1 / total_time if total_time > 0 else 0

            self._update_fps_stats(pure_inference_fps)

            cv2.putText(depth_color, f'Inference: {pure_inference_fps:.1f} FPS', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_color, f'Pipeline: {pipeline_fps:.1f} FPS', (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            self.error_count = 0

            return depth_map, depth_color, pure_inference_fps

        except Exception as e:
            self.error_count += 1
            print(f"Error in depth model inference: {e}")

            frame_height, frame_width = frame.shape[:2]
            error_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Depth Error", (50, frame_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return None, error_frame, 0

    def _update_fps_stats(self, fps):
        """Update FPS statistics and print average every 30 seconds."""
        self.fps_values.append(fps)
        current_time = time.time()

        # Print average FPS every 30 seconds
        if current_time - self.fps_calc_start_time >= 30:
            avg_fps = sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0
            # Uncomment for FPS logging: print(f"Average depth FPS: {avg_fps:.2f}")
            self.fps_values = []
            self.fps_calc_start_time = current_time

    def should_reinitialize(self):
        """Check if model should be reinitialized due to errors."""
        return self.error_count >= MAX_CONSECUTIVE_ERRORS

    def attempt_recovery(self):
        """Attempt to recover from persistent errors."""
        try:
            print("Multiple errors in depth detection, attempting to reinitialize model...")

            # Release GPU memory
            # Skip expensive CUDA cache clearing

            # Reinitialize the model
            if self.initialize_model():
                self.error_count = 0
                print("Depth model reinitialized successfully")
                return True
            else:
                print("Failed to reinitialize depth model")
                return False

        except Exception as e:
            print(f"Error during depth model recovery: {e}")
            return False

    def get_model_info(self):
        """Get information about the current model configuration."""
        return {
            'encoder': self.encoder,
            'device': self.device,
            'config': self.depth_configs[self.encoder],
            'error_count': self.error_count,
            'model_loaded': self.model is not None
        }


def run_depth_detection_thread(engine, frame_source, result_callback,
                              thread_coordinator, running_flag):
    """
    Run depth detection in a dedicated thread.

    Args:
        engine: DepthDetectionEngine instance
        frame_source: Function that returns the current frame
        result_callback: Function called with (depth_map, depth_visualization)
        thread_coordinator: ThreadCoordinator instance for monitoring
        running_flag: Shared flag indicating if thread should continue
    """
    thread_name = "depth_detection"
    thread_coordinator.register_thread(thread_name)
    print("Depth detection thread started.")

    while running_flag[0]:
        try:
            thread_coordinator.heartbeat(thread_name)

            frame = frame_source()

            if frame is None:
                time.sleep(0.1)
                continue

            depth_map, depth_visualization, fps = engine.process_frame(frame)

            result_callback(depth_map, depth_visualization)

            if engine.should_reinitialize():
                if engine.attempt_recovery():
                    thread_coordinator.record_restart(thread_name)
                else:
                    thread_coordinator.increment_error(thread_name)

            time.sleep(0.01)

        except Exception as e:
            thread_coordinator.increment_error(thread_name)
            print(f"ERROR in depth detection thread: {e}")
            time.sleep(1)