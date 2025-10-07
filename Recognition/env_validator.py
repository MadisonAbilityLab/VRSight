"""
Simple environment variable validation.
"""

import os
from simple_logger import log_error, log_info

def validate_environment():
    """
    Validate required environment variables at startup.

    Returns:
        bool: True if all required variables are present
    """
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
    ]

    optional_vars = [
        'DEBUG'
    ]

    missing_required = []
    missing_optional = []

    # Check required variables
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)

    # Check optional variables
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)

    # Report results
    if missing_required:
        log_error(f"Missing required environment variables: {', '.join(missing_required)}")
        log_error("Please check .env.example for setup instructions")
        return False

    if missing_optional:
        log_info(f"Optional environment variables not set: {', '.join(missing_optional)}")

    log_info("Environment validation passed")
    return True