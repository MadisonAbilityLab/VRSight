"""
Simple model file validation that model file exists.
"""

import os
from settings import YOLO_MODEL_PATH
from simple_logger import log_error, log_info

def validate_model_files():
    """
    Validate that required model files exist.

    Returns:
        bool: True if all required model files exist
    """
    model_files = [
        YOLO_MODEL_PATH,
    ]

    missing_files = []

    # Check each model file
    for model_path in model_files:
        if not os.path.exists(model_path):
            missing_files.append(model_path)

    # Report results
    if missing_files:
        log_error(f"Missing model files: {', '.join(missing_files)}")
        log_error("Please ensure model files are downloaded and paths are correct")
        return False

    log_info("Model file validation passed")
    return True