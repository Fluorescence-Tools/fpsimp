# Local ColabFold Integration Setup

This guide explains how to configure the AVNN-Web-FP worker to use your local ColabFold installation instead of the queue-based system.

## Overview

The worker now runs `colabfold_batch` directly instead of using a separate watcher container. This provides:
- Direct control over ColabFold execution
- Better error handling and logging
- Simpler architecture (no queue system)
- GPU support through environment variables

## Prerequisites

1. **Local ColabFold Installation**: You must have ColabFold installed at `/opt/localcolabfold/colabfold-conda/bin/colabfold_batch`

2. **Docker Desktop Configuration**: The ColabFold directory must be shared with Docker

## Setup Steps

### 1. Verify ColabFold Installation

```bash
# Check that colabfold_batch exists and is executable
ls -la /opt/localcolabfold/colabfold-conda/bin/colabfold_batch

# Test colabfold_batch
/opt/localcolabfold/colabfold-conda/bin/colabfold_batch --help
```

### 2. Configure Docker Desktop File Sharing

1. Open Docker Desktop
2. Go to **Settings** → **Resources** → **File Sharing**
3. Add `/opt/localcolabfold` to the shared directories list
4. Click **Apply & Restart**

### 3. Update Environment Variables

Create or update `.env` file in the web-app directory:

```bash
# Set ColabFold path (matches the Docker mount)
COLABFOLD_PATH=/opt/localcolabfold/colabfold-conda/bin

# Optional: Set GPU device for ColabFold
# COLABFOLD_GPU=0
```

### 4. Build and Start Services

```bash
# Build the Docker image
docker-compose build

# Start all services
docker-compose up -d

# Check worker logs
docker-compose logs -f worker
```

## How It Works

### Direct Execution

The worker now calls the existing `run_colabfold()` function from `fpsim/utils.py` directly:

```python
from fpsim.utils import run_colabfold

af_pdb_path = run_colabfold(
    fasta=Path(str(getattr(cfg, 'fasta'))),
    out_dir=cf_out,
    extra_args=colabfold_args,
    gpu=getattr(cfg, 'gpu', None)
)
```

### GPU Support

GPU selection is handled through the `gpu` parameter in PipelineConfig:

- Sets `CUDA_VISIBLE_DEVICES` environment variable
- Passed to `colabfold_batch` subprocess
- Configurable through web UI or CLI

### Error Handling

- Direct subprocess execution with proper error capture
- Detailed logging in worker logs
- Graceful fallback when ColabFold fails

## Troubleshooting

### Permission Denied

```bash
# Fix Docker mount permissions
sudo chmod -R 755 /opt/localcolabfold
```

### ColabFold Not Found

```bash
# Check worker container
docker-compose exec worker ls -la /opt/localcolabfold/colabfold-conda/bin/

# Check environment variable
docker-compose exec worker printenv | grep COLABFOLD
```

### GPU Not Available

```bash
# Check GPU visibility in container
docker-compose exec worker nvidia-smi

# Check CUDA_VISIBLE_DEVICES
docker-compose exec worker printenv | grep CUDA
```

### Permission Issues with File Sharing

1. In Docker Desktop, ensure `/opt/localcolabfold` is added to File Sharing
2. Restart Docker Desktop after adding the path
3. Verify the mount works:
   ```bash
   docker-compose exec worker ls -la /opt/localcolabfold/
   ```

## Configuration Options

### ColabFold Parameters

All ColabFold parameters are supported through the web interface:

- **Model Type**: `alphafold2`, `alphafold2_ptm`, `alphafold2_multimer_v2`, etc.
- **Number of Models**: 1-5
- **MSA Mode**: `mmseqs2_uniref_env`, `single_sequence`, etc.
- **Templates**: Enable/disable template usage
- **Relax**: Amber relaxation with GPU support

### GPU Configuration

Set GPU device in web UI or environment:

```bash
# Use specific GPU
COLABFOLD_GPU=0

# Use multiple GPUs (comma-separated)
COLABFOLD_GPU=0,1
```

## Migration from Queue System

If you were previously using the queue-based system:

1. **No queue directory needed**: The `/app/cf_queue` mount is no longer used
2. **No watcher container**: Remove any ColabFold watcher services
3. **Direct execution**: Worker handles ColabFold directly
4. **Better logging**: All ColabFold output appears in worker logs

## Testing

```bash
# Test with a simple FASTA
echo ">test
MKWVTFISLLFLFSSAYSRGVFRR" > test.fasta

# Submit through web interface or API
curl -X POST http://localhost:8000/api/submit \
  -F "fasta=@test.fasta" \
  -F "run_colabfold=true" \
  -F "colabfold_args=--num-models 1"
```

Monitor the job progress at `http://localhost:8000/results/<job_id>` and check worker logs for ColabFold output.
