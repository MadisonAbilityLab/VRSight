from config import get_config

# Load configuration
_config = get_config()

# ============================================================================
# DEBUG SETTINGS
# ============================================================================
DEBUG = _config.debug

# ============================================================================
# CAMERA AND DISPLAY SETTINGS
# ============================================================================
CAM_WIDTH = _config.camera.width
CAM_HEIGHT = _config.camera.height
WEBCAM_INDEX = _config.camera.webcam_index

# ============================================================================
# MODEL PATHS AND DEVICE CONFIGURATION
# ============================================================================
YOLO_MODEL_PATH = _config.models.yolo_model_path
DEVICE = _config.models.device

# ============================================================================
# DETECTION THRESHOLDS AND PARAMETERS
# ============================================================================
# Edge detection
LINE_TOLERANCE = _config.detection.line_tolerance
EDGE_TOLERANCE = _config.detection.edge_tolerance
HOUGH_THRESHOLD = _config.detection.hough_threshold
MIN_LINE_LENGTH = _config.detection.min_line_length
MAX_LINE_GAP = _config.detection.max_line_gap
LINE_STRAIGHTNESS_THRESHOLD = _config.detection.line_straightness_threshold

# Proximity thresholds (pixels)
HAND_PROXIMITY_THRESHOLD = _config.detection.hand_proximity_threshold
INTERACTION_PROXIMITY_THRESHOLD = _config.detection.interaction_proximity_threshold
BBOX_EXPANSION_PIXELS = _config.detection.bbox_expansion_pixels

# ============================================================================
# THREADING AND PERFORMANCE SETTINGS
# ============================================================================
# Thread monitoring
THREAD_HEARTBEAT_TIMEOUT = _config.performance.thread_heartbeat_timeout
THREAD_ERROR_THRESHOLD = _config.performance.thread_error_threshold
MAX_THREAD_RESTARTS = _config.performance.max_thread_restarts
THREAD_SHUTDOWN_TIMEOUT = _config.performance.thread_shutdown_timeout

# Performance monitoring
MAX_STAT_HISTORY = _config.performance.max_stat_history
MAX_INFERENCE_TIME = _config.performance.max_inference_time

# Queue settings
QUEUE_MAX_SIZE = _config.performance.queue_max_size

# Memory management
MEMORY_CLEANUP_THRESHOLD_MB = _config.performance.memory_cleanup_threshold_mb
MEMORY_CLEANUP_INTERVAL = _config.performance.memory_cleanup_interval

# ============================================================================
# COOLDOWN AND RATE LIMITING SETTINGS
# ============================================================================
# Special label cooldowns (seconds)
COOLDOWNS = _config.rate_limiting.cooldowns

# Category cooldowns
COOLDOWN_INTERACTABLES = _config.rate_limiting.cooldown_interactables
COOLDOWN_USER_SAFETY = _config.rate_limiting.cooldown_user_safety
COOLDOWN_INFORMATIONAL = _config.rate_limiting.cooldown_informational

# GPT rate limiting
GPT_MIN_REQUEST_INTERVAL = _config.rate_limiting.gpt_min_request_interval

# Speech settings
MIN_SILENCE_PERIOD = _config.rate_limiting.min_silence_period
USER_COMMAND_MAX_DURATION = _config.rate_limiting.user_command_max_duration
SPEECH_QUEUE_STALE_TIMEOUT = _config.rate_limiting.speech_queue_stale_timeout

# ============================================================================
# AUDIO AND WEBSOCKET SETTINGS
# ============================================================================
WEBSOCKET_HOST = _config.network.websocket_host
WEBSOCKET_PORT = _config.network.websocket_port

# PlayCanvas integration
ASSET_ID = _config.network.asset_id
PROJECT_ID = _config.network.project_id
FOLDER_ID = _config.network.folder_id
PLAYCANVAS_CAMERA_HEIGHT = _config.network.playcanvas_camera_height

# ============================================================================
# DETECTION CATEGORIES AND LABELS
# ============================================================================
# Edge detection labels (hands/controllers)
EDGE_LABELS = ['hand', 'controller']

# Special labels for automatic processing
SPECIAL_LABELS = ['progress bar', 'guardian', 'dashboard', 'out of bounds']

# Interaction-relevant object labels
INTERACTION_LABELS = [
    'button', 'interactable', 'portal', 'menu',
    'sign-text', 'ui-text', 'sign-graphic', 'ui-graphic',
    'progress bar'
]

# ============================================================================
# DEPTH PROCESSING SETTINGS
# ============================================================================
# World coordinate mapping
MIN_DEPTH_DISTANCE = 1.0       # meters
MAX_DEPTH_DISTANCE = 80.0      # meters
DEPTH_SCALE_FACTOR = 0.75

# Field of view
DEFAULT_FOV = 90               # degrees

# ============================================================================
# COLOR DETECTION RANGES (HSV)
# ============================================================================
# Pointer color ranges for edge detection
POINTER_COLORS = {
    'green': {
        'lower': (40, 30, 150),
        'upper': (85, 255, 255)
    },
    'teal': {
        'lower': (80, 10, 80),
        'upper': (120, 255, 255)
    },
    'white': {
        'lower': (20, 10, 180),
        'upper': (60, 60, 255)
    }
}

# ============================================================================
# ERROR HANDLING AND RETRY SETTINGS
# ============================================================================
MAX_CONSECUTIVE_ERRORS = 5
MAX_CAMERA_RETRIES = 5
OCR_CACHE_TTL = 30             # seconds
OCR_CACHE_MAX_SIZE = 100

# API failure handling
MAX_FAILED_SPEECH_ATTEMPTS = 5