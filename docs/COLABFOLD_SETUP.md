# ColabFold Setup for macOS

## Overview

On macOS, ColabFold runs on **bare metal** (host system) rather than inside Docker containers, because:
- Docker on macOS doesn't support GPU passthrough
- ARM64 architecture cannot access NVIDIA GPUs even with emulation
- Native execution provides better performance

## Installation

### 1. Install ColabFold on macOS Host

```bash
# Install localcolabfold
cd /opt
sudo mkdir localcolabfold
sudo chown $USER localcolabfold
cd localcolabfold

# Download and install
curl -O https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabfold_macos.sh
bash install_colabfold_macos.sh
```

### 2. Verify Installation

```bash
# Test ColabFold
/opt/localcolabfold/colabfold-conda/bin/colabfold_batch --help

# Should show ColabFold help text
```

### 3. Configure Environment

Edit `/Users/tpeulen/dev/avnn/web-app/.env`:

```bash
# Set to your actual ColabFold installation path
COLABFOLD_PATH=/opt/localcolabfold/colabfold-conda/bin
```

### 4. Update docker-compose.yml

The worker container needs access to host ColabFold:

```yaml
worker:
  volumes:
    # Mount host ColabFold (read-only)
    - /opt/localcolabfold:/opt/localcolabfold:ro
  environment:
    - COLABFOLD_PATH=/opt/localcolabfold/colabfold-conda/bin
```

**Important:** Update the mount path if you installed ColabFold in a different location.

## How It Works

1. **Worker Container**: Runs IMP/PMI simulations inside Docker
2. **ColabFold**: Runs on macOS host via `subprocess.run()`
3. **Communication**: Worker calls host binary through mounted volume

```
┌─────────────────────────────────────┐
│  macOS Host                         │
│  ┌───────────────────────────────┐  │
│  │  ColabFold (bare metal)       │  │
│  │  /opt/localcolabfold/...      │  │
│  └───────────────────────────────┘  │
│           ▲                          │
│           │ subprocess call          │
│           │                          │
│  ┌────────┴──────────────────────┐  │
│  │  Docker Worker Container      │  │
│  │  - Mounted volume access      │  │
│  │  - Calls colabfold_batch      │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Usage

### Submit Job with ColabFold

```python
# Job parameters
{
    "run_colabfold": true,
    "colabfold_args": "--num-models 1 --num-recycle 3",
    "use_gpu_relax": false,  # GPU relax not available on macOS
    ...
}
```

### Check Execution

```bash
# Monitor worker logs
docker compose logs -f worker

# Should see output like:
# [colabfold] Using ColabFold from: /opt/localcolabfold/colabfold-conda/bin
# [colabfold] Running: /opt/localcolabfold/.../colabfold_batch ...
```

## Troubleshooting

### ColabFold Not Found

```bash
# Error: colabfold_batch not found
```

**Solution:**
1. Check installation: `ls -la /opt/localcolabfold/colabfold-conda/bin/colabfold_batch`
2. Update `COLABFOLD_PATH` in `.env`
3. Update mount path in `docker-compose.yml`
4. Restart worker: `docker compose restart worker`

### Permission Issues

```bash
# Error: Permission denied
```

**Solution:**
```bash
# Make ColabFold executable
chmod +x /opt/localcolabfold/colabfold-conda/bin/colabfold_batch

# Or install in user directory
cd ~/
mkdir localcolabfold
# ... install there instead
```

### Different Installation Location

If you installed ColabFold elsewhere (e.g., `/usr/local/colabfold`):

1. Update `.env`:
   ```bash
   COLABFOLD_PATH=/usr/local/colabfold/bin
   ```

2. Update `docker-compose.yml`:
   ```yaml
   volumes:
     - /usr/local/colabfold:/usr/local/colabfold:ro
   environment:
     - COLABFOLD_PATH=/usr/local/colabfold/bin
   ```

3. Restart worker:
   ```bash
   docker compose restart worker
   ```

## Performance Notes

- **CPU-only**: ColabFold on macOS runs on CPU (no GPU acceleration)
- **Speed**: ~10-30x slower than GPU for structure prediction
- **Memory**: Requires 8-16GB RAM for typical proteins
- **Recommended**: Use pre-computed structures or external GPU server for production

## Alternative: External ColabFold

For faster predictions, consider:
1. **Google Colab**: Use official ColabFold notebook, download PDB
2. **Remote Server**: Deploy ColabFold on Linux + NVIDIA GPU
3. **Pre-computed**: Use existing AlphaFold structures from databases

Pass pre-computed PDB via `af_pdb` parameter:
```python
{
    "run_colabfold": false,
    "af_pdb": "/path/to/existing_structure.pdb",
    ...
}
```
