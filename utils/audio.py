"""
Browser-side audio helpers for LazyResident.
Handles persisting user-recorded audio from the browser to
temporary files that can be sent to the Gemini transcription API.
"""
from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from .config import AUDIO_DIR

logger = logging.getLogger(__name__)

@contextmanager
def temporary_audio_file(data: bytes) -> Generator[Path, None, None]:
    """Persist raw audio bytes to disk long enough for transcription."""
    if not data:
        raise ValueError("No audio data provided for transcription.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=AUDIO_DIR, suffix=".wav", delete=False) as temp:
        temp.write(data)
        temp.flush()
        temp_path = Path(temp.name)

    logger.debug("Temporary audio saved: %s", temp_path.name)

    try:
        yield temp_path
    finally:
        try:
            temp_path.unlink()
            logger.debug("Temporary audio removed: %s", temp_path.name)
        except FileNotFoundError:
            pass
        except PermissionError as exc:
            logger.warning("Temporary audio locked (%s): %s", temp_path, exc)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to remove temp audio %s: %s", temp_path, exc)
