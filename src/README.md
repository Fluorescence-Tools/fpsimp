# FPSIMP Source Code

This directory contains the main application source code for FPSIMP.

## Files

- **`app.py`** - Main Flask application with REST API endpoints
- **`celery_worker.py`** - Celery worker configuration for background jobs
- **`config.py`** - Centralized configuration management
- **`__init__.py`** - Package initialization and exports

## Usage

### Development
```bash
# Start the Flask application
cd src
python app.py

# Start Celery worker (in separate terminal)
celery -A src.celery worker --loglevel=info
```

### Docker
The source code is mounted in the Docker containers at `/app/src/` and executed from there.

## Architecture

```
src/
├── app.py           # Flask web server and API endpoints
├── celery_worker.py # Celery configuration and worker setup
├── config.py        # Environment variables and settings
└── __init__.py      # Package exports
```

## Imports

The source code uses relative imports for the configuration:
```python
from .config import config
```

This ensures the package works correctly when moved to the `src/` directory.
