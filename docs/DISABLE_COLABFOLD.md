# DISABLE_COLABFOLD Environment Variable

## Overview
The `DISABLE_COLABFOLD` environment variable allows you to completely disable ColabFold functionality in the AVNN web application. When enabled, users must upload structure files with pLDDT scores instead of relying on ColabFold for structure prediction.

## Usage

### Enable ColabFold (Default)
```bash
DISABLE_COLABFOLD=false
```
- Users can upload FASTA files
- ColabFold prediction is available
- All UI elements are visible
- ColabFold services (master/worker) are started

### Disable ColabFold
```bash
DISABLE_COLABFOLD=true
```
- Users must upload structure files (PDB/mmCIF) with pLDDT scores
- FASTA-only uploads are not supported
- ColabFold UI elements are hidden
- Job submission forces `run_colabfold=False`
- ColabFold services are NOT started (resource savings)

## Deployment

### With ColabFold (Default)
```bash
# Method 1: Using deploy script
./deploy.sh

# Method 2: Manual
docker-compose --profile colabfold up -d --build
```

### Without ColabFold (Resource Optimized)
```bash
# Method 1: Using deploy script (recommended)
DISABLE_COLABFOLD=true ./deploy.sh

# Method 2: Manual
DISABLE_COLABFOLD=true docker-compose up -d --build

# Method 3: Environment file
echo "DISABLE_COLABFOLD=true" >> .env
docker-compose up -d --build
```

## Resource Savings

When `DISABLE_COLABFOLD=true`:
- **No ColabFold containers**: Saves significant CPU/RAM
- **No GPU usage**: ColabFold worker containers not started
- **Faster startup**: Fewer services to initialize
- **Smaller footprint**: Reduced Docker image requirements

## Service Differences

| Service | ColabFold Enabled | ColabFold Disabled |
|---------|------------------|-------------------|
| web | ✅ | ✅ |
| worker | ✅ | ✅ |
| monitor | ✅ | ✅ |
| redis | ✅ | ✅ |
| colabfold-master | ✅ | ❌ |
| colabfold-worker | ✅ | ❌ |

## Implementation Details

### Backend Changes
1. **Environment Variable**: Added to Flask app config in `app.py`
2. **API Endpoint**: `/api/config` returns ColabFold status to frontend
3. **Job Submission**: Forces `run_colabfold=False` when disabled
4. **Pipeline Logic**: Skips ColabFold task chaining
5. **Intelligent Downloader**: Uses latest AlphaFold versions when available

### Frontend Changes
1. **Configuration Fetch**: App fetches config on initialization
2. **UI Module**: `colabfold-ui.js` handles element visibility
3. **Hidden Elements**:
   - ColabFold prediction section
   - Multimer sequence display
   - Memory fraction, GPU device, model type inputs
   - Additional arguments field
   - FASTA file upload section

4. **User Guidance**:
   - Upload section text changes to "Upload Structure with pLDDT Scores"
   - Warning message explains structure requirements
   - Only UniProt names allowed for remote downloads (simplified approach)

### Docker Configuration
1. **docker-compose.yml**: Uses profiles to conditionally start ColabFold services
2. **deploy.sh**: Smart deployment script that checks `DISABLE_COLABFOLD`
3. **.env file**: Added variable with documentation

## User Experience Changes

### When ColabFold is Disabled:
- Upload section requires structure files with pLDDT
- ColabFold parameters section is hidden
- Multimer sequence display is hidden
- Clear warning about structure requirements
- Jobs run directly without ColabFold prediction
- Only UniProt names allowed for remote downloads (simplified approach)
- Faster application startup

### When ColabFold is Enabled:
- Full functionality preserved
- FASTA uploads work normally
- All UI elements visible
- ColabFold prediction available
- PDB ID and UniProt name downloads supported
- ColabFold services run in background

## Benefits
- **Reduced Resource Usage**: No ColabFold containers needed
- **Faster Deployment**: Smaller Docker footprint
- **Simplified Workflow**: Direct structure processing
- **Backward Compatibility**: Existing code preserved
- **Flexible Deployment**: Easy toggle via environment variable

## Troubleshooting

### ColabFold Services Not Starting
```bash
# Check if DISABLE_COLABFOLD is set
echo $DISABLE_COLABFOLD

# Force enable ColabFold
unset DISABLE_COLABFOLD
./deploy.sh
```

### UI Elements Still Visible
```bash
# Restart web service to pick up environment changes
docker-compose restart web
```

### Structure Upload Issues
When ColabFold is disabled, ensure:
- Files are PDB/mmCIF format
- Files contain pLDDT scores in B-factor column
- Use UniProt names for remote downloads (e.g., P42212) - system handles AlphaFold automatically
