# Worker Scripts

This directory contains scripts for ColabFold job processing and management.

## Files

### Core ColabFold Scripts
- **`colabfold_worker.py`** - Simple ColabFold worker that processes individual jobs
- **`colabfold_master.py`** - ColabFold master service that manages job distribution
- **`cf_watch_simple.py`** - Simple job watcher for CPU-based ColabFold processing

## Usage

These scripts are used by the Docker containers defined in `Dockerfile.colabfold.worker` and `Dockerfile.colabfold.master`. They are not intended to be run directly.

### Docker Integration
- The worker scripts are copied to `/colabfold/worker/` in the worker container
- The master script is copied to `/app/` in the master container
- Scripts are executed via Docker exec commands from the main application

## Architecture

```
Main App (app.py) → Docker Exec → Worker Container → colabfold_worker.py
                                    ↓
                              ColabFold Processing
                                    ↓
                              Results → Main App
```

## Removed Files

- `colabfold_client.py` - gRPC client (not used in current implementation)
