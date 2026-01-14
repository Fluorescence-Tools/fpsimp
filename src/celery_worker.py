#!/usr/bin/env python3
"""
Celery worker configuration for FPSIMP
"""
import os
import sys
from pathlib import Path

# Add src directory to Python path (standalone deployment)
sys.path.append(str(Path(__file__).parent))

from app import celery, app

if __name__ == '__main__':
    with app.app_context():
        celery.start()
