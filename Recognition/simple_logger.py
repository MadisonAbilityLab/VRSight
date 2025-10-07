import logging
import sys
from settings import DEBUG

def setup_logger(name="vr_ai"):
    """
    Set up simple logger with console output.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set level based on DEBUG setting
    level = logging.DEBUG if DEBUG else logging.INFO
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Simple format
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

# Global logger instance
logger = setup_logger()

def log_error(message):
    """Log error message."""
    logger.error(message)

def log_info(message):
    """Log info message."""
    logger.info(message)

def log_debug(message):
    """Log debug message (only if DEBUG=True)."""
    logger.debug(message)