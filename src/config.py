"""
Centralized configuration for FPSIMP application.

This module loads and validates all environment variables used throughout the application.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base configuration
class Config:
    # Application settings
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    
    # File storage
    UPLOAD_FOLDER: Path = Path(os.getenv('UPLOAD_FOLDER', 'uploads')).resolve()
    RESULTS_FOLDER: Path = Path(os.getenv('RESULTS_FOLDER', 'results')).resolve()
    MAX_CONTENT_LENGTH: int = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB default
    
    # Celery configuration (new lowercase format)
    broker_url: str = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
    result_backend: str = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
    # Add old uppercase names for backward compatibility
    CELERY_BROKER_URL: str = broker_url
    CELERY_RESULT_BACKEND: str = result_backend
    
    # ColabFold service
    COLABFOLD_SERVICE: str = os.getenv('COLABFOLD_SERVICE', 'colabfold:50051')
    
    # Legacy AVNN settings (deprecated - not used in standalone deployment)
    # AVNN_PATH: Optional[Path] = Path(os.getenv('AVNN_PATH', '../avnn')).resolve() if os.getenv('AVNN_PATH') else None
    
    # Web server settings
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', '5000'))
    
    # GPU settings
    GPU_ENABLED: bool = os.getenv('GPU_ENABLED', 'true').lower() == 'true'
    NVIDIA_VISIBLE_DEVICES: str = os.getenv('NVIDIA_VISIBLE_DEVICES', 'all')
    
    # Job cleanup settings
    CLEANUP_JOBS_AFTER_DAYS: int = int(os.getenv('CLEANUP_JOBS_AFTER_DAYS', '7'))

    # Celery task routing
    CELERY_TASK_ROUTES = {
        'run_colabfold_task': {'queue': 'imp_queue'},
        'run_fpsim_job': {'queue': 'imp_queue'},
        'run_pipeline_task': {'queue': 'imp_queue'}
    }
    
    # Email notification settings (SMTP)
    SMTP_SERVER: str = os.getenv('SMTP_SERVER', '')
    SMTP_PORT: int = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USER: str = os.getenv('SMTP_USER', '')
    SMTP_PASS: str = os.getenv('SMTP_PASS', '')
    SMTP_FROM: str = os.getenv('SMTP_FROM', 'fpsimp@localhost')
    SMTP_ENABLED: bool = os.getenv('SMTP_ENABLED', 'false').lower() == 'true'
    
    def __init__(self):
        # Create directories if they don't exist
        self.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        self.RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Create config instance
config = Config()

# For backward compatibility
def get_config() -> Config:
    return config

def validate_config() -> None:
    """Validate configuration and raise exceptions for invalid settings."""
    # Check if required directories exist
    if not config.UPLOAD_FOLDER.exists():
        raise FileNotFoundError(f"Upload directory not found: {config.UPLOAD_FOLDER}")
    if not config.RESULTS_FOLDER.exists():
        raise FileNotFoundError(f"Results directory not found: {config.RESULTS_FOLDER}")
    
    # Using local fpsim module - fully standalone deployment

# Validate configuration on import
try:
    validate_config()
except Exception as e:
    print(f"Configuration error: {e}")
    if config.DEBUG:
        print("Continuing in debug mode...")
    else:
        raise
