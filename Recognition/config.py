"""
Configuration management handling device-specific configurations with validation.
"""

import os
import json
import torch
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import logging


class DeviceType(Enum):
    """Device types for optimization"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class CameraConfig:
    """Camera configuration"""
    width: int = 640
    height: int = 640
    webcam_index: int = 1
    max_retries: int = 5
    warmup_time: float = 1.0


@dataclass
class ModelConfig:
    """Model configuration"""
    yolo_model_path: str = 'weights/best.pt'
    depth_encoder: str = 'vits'  # vits, vitb, vitl, vitg
    device: str = 'auto'
    model_precision: str = 'fp16'  # fp32, fp16, int8 - fp16 default for performance


@dataclass
class DetectionConfig:
    """Detection thresholds and parameters"""
    # Edge detection
    line_tolerance: int = 20
    edge_tolerance: int = 30
    hough_threshold: int = 25
    min_line_length: int = 40
    max_line_gap: int = 25
    line_straightness_threshold: float = 0.85

    # Proximity thresholds
    hand_proximity_threshold: int = 15
    interaction_proximity_threshold: int = 60
    bbox_expansion_pixels: int = 25


@dataclass
class PerformanceConfig:
    """Performance and threading configuration"""
    # Threading
    thread_heartbeat_timeout: int = 10
    thread_error_threshold: int = 5
    max_thread_restarts: int = 10
    thread_shutdown_timeout: float = 5.0

    # Performance monitoring
    max_stat_history: int = 100
    max_inference_time: float = 2.0
    queue_max_size: int = 10

    # Memory management
    memory_cleanup_threshold_mb: int = 1000
    memory_cleanup_interval: int = 60


@dataclass
class RateLimitingConfig:
    """Rate limiting and cooldown configuration"""
    # Special label cooldowns
    cooldowns: Dict[str, int] = field(default_factory=lambda: {
        "guardian": 5,
        "out of bounds": 5,
        "progress bar": 15,
        "dashboard": 5
    })

    # Category cooldowns
    cooldown_interactables: int = 30
    cooldown_user_safety: int = 5
    cooldown_informational: int = 60

    # GPT rate limiting
    gpt_min_request_interval: int = 10

    # Speech settings
    min_silence_period: int = 1
    user_command_max_duration: int = 120
    speech_queue_stale_timeout: int = 60


@dataclass
class NetworkConfig:
    """Network and WebSocket configuration"""
    websocket_host: str = "localhost"
    websocket_port: int = 8765

    # PlayCanvas integration
    asset_id: str = "207906643"
    project_id: str = "1233172"
    folder_id: str = "206527486"
    playcanvas_camera_height: int = 2


@dataclass
class SystemConfig:
    """Complete system configuration"""
    debug: bool = False
    camera: CameraConfig = field(default_factory=CameraConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    rate_limiting: RateLimitingConfig = field(default_factory=RateLimitingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)


class ConfigValidator:
    """Validates configuration values and dependencies"""

    @staticmethod
    def validate_camera_config(config: CameraConfig) -> List[str]:
        """Validate camera configuration"""
        errors = []

        if config.width <= 0 or config.height <= 0:
            errors.append("Camera width and height must be positive")

        if config.webcam_index < 0:
            errors.append("Webcam index must be non-negative")

        if config.max_retries < 1:
            errors.append("Max retries must be at least 1")

        return errors

    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """Validate model configuration"""
        errors = []

        if not os.path.exists(config.yolo_model_path):
            errors.append(f"YOLO model file not found: {config.yolo_model_path}")

        valid_encoders = ['vits', 'vitb', 'vitl', 'vitg']
        if config.depth_encoder not in valid_encoders:
            errors.append(f"Invalid depth encoder: {config.depth_encoder}")

        valid_precisions = ['fp32', 'fp16', 'int8']
        if config.model_precision not in valid_precisions:
            errors.append(f"Invalid model precision: {config.model_precision}")

        return errors

    @staticmethod
    def validate_detection_config(config: DetectionConfig) -> List[str]:
        """Validate detection configuration"""
        errors = []

        if config.line_tolerance < 0:
            errors.append("Line tolerance must be non-negative")

        if not 0 <= config.line_straightness_threshold <= 1:
            errors.append("Line straightness threshold must be between 0 and 1")

        if config.hough_threshold <= 0:
            errors.append("Hough threshold must be positive")

        return errors

    @staticmethod
    def validate_performance_config(config: PerformanceConfig) -> List[str]:
        """Validate performance configuration"""
        errors = []

        if config.thread_heartbeat_timeout <= 0:
            errors.append("Thread heartbeat timeout must be positive")

        if config.memory_cleanup_threshold_mb <= 0:
            errors.append("Memory cleanup threshold must be positive")

        if config.queue_max_size <= 0:
            errors.append("Queue max size must be positive")

        return errors


class ConfigManager:
    """Manages configurations with validation"""

    def __init__(self, debug: bool = None):
        self.debug = debug if debug is not None else self._detect_debug_mode()
        self._config: Optional[SystemConfig] = None
        self._logger = logging.getLogger(__name__)

    def _detect_debug_mode(self) -> bool:
        """Auto-detect debug mode from environment variables or command line"""
        return os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes')

    def _detect_optimal_device(self) -> str:
        """Auto-detect optimal device for the current system"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_base_config(self) -> SystemConfig:
        """Get base configuration"""
        base_config = SystemConfig(debug=self.debug)

        # Auto-detect device
        base_config.models.device = self._detect_optimal_device()

        return base_config

    def _apply_debug_overrides(self, config: SystemConfig) -> SystemConfig:
        """Apply debug-specific overrides"""
        if config.debug:
            # Debug mode: more verbose logging and longer timeouts
            config.performance.max_stat_history = 200  # More debugging info
            config.performance.thread_heartbeat_timeout = 15  # Longer timeouts for debugging

        return config

    def _apply_device_optimizations(self, config: SystemConfig) -> SystemConfig:
        """Apply device-specific optimizations"""
        device = config.models.device

        if device == "cuda":
            # CUDA optimizations
            config.performance.memory_cleanup_interval = 45  # More frequent cleanup
            config.performance.queue_max_size = 15  # Larger queues for GPU
            config.performance.max_inference_time = 1.5  # Tighter timeout for CUDA

        elif device == "mps":
            # Apple Silicon optimizations
            config.performance.memory_cleanup_interval = 60
            config.models.model_precision = 'fp16'  # Native MPS precision

        else:  # CPU
            # CPU optimizations
            config.performance.queue_max_size = 5  # Smaller queues
            config.performance.max_inference_time = 5.0  # More lenient timeouts

        return config

    def load_config(self, config_file: Optional[str] = None) -> SystemConfig:
        """Load and validate configuration"""
        if self._config is not None:
            return self._config

        # Start with base configuration
        config = self._get_base_config()

        # Load from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    # Apply file overrides (implementation would merge configs)
                    self._logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self._logger.warning(f"Failed to load config file: {e}")

        # Apply debug-specific settings
        config = self._apply_debug_overrides(config)

        # Apply device-specific optimizations
        config = self._apply_device_optimizations(config)

        # Validate configuration
        errors = self.validate_config(config)
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)

        self._config = config
        debug_status = "debug" if config.debug else "normal"
        self._logger.info(f"Configuration loaded in {debug_status} mode on {config.models.device}")

        return config

    def validate_config(self, config: SystemConfig) -> List[str]:
        """Validate entire configuration"""
        all_errors = []

        all_errors.extend(ConfigValidator.validate_camera_config(config.camera))
        all_errors.extend(ConfigValidator.validate_model_config(config.models))
        all_errors.extend(ConfigValidator.validate_detection_config(config.detection))
        all_errors.extend(ConfigValidator.validate_performance_config(config.performance))

        return all_errors

    def get_config(self) -> SystemConfig:
        """Get the current configuration, loading if necessary"""
        if self._config is None:
            return self.load_config()
        return self._config

    def save_config(self, filepath: str):
        """Save current configuration to file"""
        if self._config is None:
            raise ValueError("No configuration loaded")

        # Convert to JSON-serializable format
        config_dict = {
            'debug': self._config.debug,
            'camera': self._config.camera.__dict__,
            'models': self._config.models.__dict__,
            'detection': self._config.detection.__dict__,
            'performance': self._config.performance.__dict__,
            'rate_limiting': self._config.rate_limiting.__dict__,
            'network': self._config.network.__dict__
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        self._logger.info(f"Configuration saved to {filepath}")


# Global instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> SystemConfig:
    """Get the current system configuration"""
    return get_config_manager().get_config()


# Backward compatibility - expose individual configuration sections
def get_camera_config() -> CameraConfig:
    return get_config().camera

def get_model_config() -> ModelConfig:
    return get_config().models

def get_detection_config() -> DetectionConfig:
    return get_config().detection

def get_performance_config() -> PerformanceConfig:
    return get_config().performance