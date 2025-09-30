"""
Configuration settings for LazyResident
Streamlit deployment-ready configuration without local dependencies
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TEMP_DIR = PROJECT_ROOT / "temp"
AUDIO_DIR = TEMP_DIR / "audio"

# Create directories if they don't exist
TEMP_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# Audio Configuration
SAMPLE_RATE = 16000  # Hz
AUDIO_FORMAT = "wav"
CHUNK_SIZE = 1024

# UI Configuration
STREAMLIT_PAGE_TITLE = "LazyResident - Medical Note Generator"
STREAMLIT_PAGE_ICON = str(PROJECT_ROOT / "assets" / "images" / "icon.png")

# Logging configuration
LOG_LEVEL_NAME = os.getenv("LAZYRESIDENT_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.ERROR)


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root_logger.addHandler(handler)
    root_logger.setLevel(LOG_LEVEL)


_configure_logging()

def get_configuration_status() -> Dict[str, Any]:
    """
    Get configuration status for system checks
    
    Returns:
        Dictionary with configuration status
    """
    return {
        "log_level": logging.getLevelName(logging.getLogger().level),
        "project_root": str(PROJECT_ROOT),
        "temp_dir": str(TEMP_DIR),
        "audio_dir": str(AUDIO_DIR),
        # "gemini_models": GEMINI_MODELS
    }

def validate_configuration() -> tuple[bool, list[str]]:
    """
    Validate essential configuration
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not TEMP_DIR.exists():
        errors.append(f"Temporary directory not accessible: {TEMP_DIR}")
    
    if not AUDIO_DIR.exists():
        errors.append(f"Audio directory not accessible: {AUDIO_DIR}")
    
    return len(errors) == 0, errors