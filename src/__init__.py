"""
FPSIMP Source Package

This package contains the main application source code.
"""

from .app import app, celery
from .config import config

__version__ = "1.0.0"
__all__ = ["app", "celery", "config"]
