# AVNN-Web-FP Deployment Guide

## Quick Start

### 1. Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available (16GB+ recommended for production)
- NVIDIA GPU with CUDA support (for ColabFold)
- Ports 5000 (web), 6379 (Redis), and 5555 (monitor) available

### 2. Initial Setup
```bash
# Clone the repository
git clone <repository-url>
cd avnn-web-app

# Edit the configuration (see Configuration section below)
nano .env
```

### 3. Deploy with Docker Compose
```bash
# Build and start all services
./run.sh

# Or manually:
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Access the Application
- **Web Interface**: http://localhost:5000
- **Job Monitor**: http://localhost:5555

## Configuration

### Environment Variables

All configuration is managed through environment variables. The application looks for these in the following order:
1. Environment variables set in `docker-compose.yml`
2. Variables in `.env` file
3. Default values from `config.py`

#### Core Configuration
```bash
# Application Settings
DEBUG=false
SECRET_KEY=your-secure-secret-key

# File Storage
UPLOAD_FOLDER=uploads
RESULTS_FOLDER=results
MAX_CONTENT_LENGTH=16777216  # 16MB

# Web Server
HOST=0.0.0.0
PORT=5000
```

#### Celery Configuration
```bash
# Redis Configuration
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

#### ColabFold Configuration
```bash
# ColabFold Service
COLABFOLD_SERVICE=colabfold:50051
COLABFOLD_PATH=/opt/localcolabfold/colabfold-conda/bin
```

#### GPU Configuration
```bash
# Enable/Disable GPU
GPU_ENABLED=true
NVIDIA_VISIBLE_DEVICES=all
```

### Configuration Files

- `.env`: Main configuration file (copy from `.env.example`)
- `config.py`: Centralized configuration module
- `CONFIGURATION.md`: Complete configuration reference

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web (Flask)   │    │ Worker (Celery) │    │  Monitor (Flower)│
│   Port: 5000    │◄──►│   Background    │◄──►│   Port: 5555    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Redis (Port: 6379)                          │
│              Queue + Cache + Job Status                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  ColabFold      │
│  (GPU-enabled)  │
└─────────────────┘
```

## Testing the Deployment

### Automated Testing
```bash
# Run the test suite
python -m pytest tests/
```

### Manual Testing
1. **Upload Test File**:
   - Use `examples/test_single.fasta` or `examples/test_multi.fasta`
   - Verify sequences are displayed correctly

2. **Submit Test Job**:
   - Use minimal parameters for quick testing:
     - Frames: 1000
     - Steps: 5
     - Disable ColabFold if not needed
   - Verify job status updates in real-time
   - Check logs for any errors

3. **Verify Results**:
   - Download generated RMF files
   - Check output directory structure
   - Verify all expected files are generated

# Application Settings
FLASK_ENV=production
FLASK_DEBUG=false
```

### Docker Compose Override
Create `docker-compose.override.yml` for custom settings:
```yaml
version: '3.8'
services:
  web:
    ports:
      - "8080:5000"  # Custom port
    environment:
      - FLASK_DEBUG=true
  
  worker:
    environment:
      - CELERY_CONCURRENCY=2  # Limit workers
```

## Monitoring and Maintenance

### Health Checks
```bash
# Application health
curl http://localhost:5000/api/health

# Redis health
docker-compose exec redis redis-cli ping

# Worker status
curl http://localhost:5555/api/workers
```

### Log Management
```bash
# View all logs
docker-compose logs

# Follow specific service
docker-compose logs -f web
docker-compose logs -f worker

# Log rotation (add to crontab)
docker-compose logs --no-color > logs/app-$(date +%Y%m%d).log
```

### Backup and Recovery
```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE

# Backup results
tar -czf backup-$(date +%Y%m%d).tar.gz results/

# Restore Redis
docker-compose exec redis redis-cli FLUSHALL
docker cp backup.rdb redis_container:/data/dump.rdb
docker-compose restart redis
```

## Performance Tuning

### Resource Limits
```yaml
# docker-compose.yml additions
services:
  web:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
  
  worker:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

### Scaling Workers
```bash
# Scale workers horizontally
docker-compose up -d --scale worker=3

# Monitor worker performance
docker stats
```

### Database Optimization
```bash
# Redis memory optimization
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## Security Considerations

### Production Deployment
1. **Reverse Proxy**: Use nginx for SSL termination
2. **Authentication**: Add user authentication system
3. **File Validation**: Enhanced FASTA validation
4. **Rate Limiting**: Implement request rate limiting
5. **Network Security**: Use Docker networks

### SSL Configuration
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   lsof -i :5000
   lsof -i :6379
   
   # Change ports in docker-compose.yml
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Reduce job parameters
   # frames: 10000 (instead of 100000)
   # workers: 1 (instead of multiple)
   ```

3. **Redis Connection Errors**
   ```bash
   # Check Redis status
   docker-compose ps redis
   
   # Restart Redis
   docker-compose restart redis
   ```

4. **Worker Not Processing Jobs**
   ```bash
   # Check worker logs
   docker-compose logs worker
   
   # Restart worker
   docker-compose restart worker
   ```

### Debug Mode
```bash
# Enable debug logging
export FLASK_DEBUG=1
export CELERY_LOG_LEVEL=DEBUG

# Run with verbose output
docker-compose up --no-daemon
```

## API Reference

### Endpoints
- `GET /` - Web interface
- `GET /api/health` - Health check
- `POST /api/upload` - Upload FASTA file
- `POST /api/submit` - Submit job
- `GET /api/status/{job_id}` - Job status
- `GET /api/results/{job_id}` - Job results
- `GET /api/download/{job_id}/{filename}` - Download file

### Response Codes
- `200` - Success
- `400` - Bad request (validation error)
- `404` - Not found
- `413` - File too large
- `500` - Internal server error

## Updates and Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose down && docker-compose up -d
```

### Database Migration
```bash
# Backup current data
docker-compose exec redis redis-cli BGSAVE

# Update schema (if needed)
# Run migration scripts

# Verify data integrity
python test_deployment.py
```

## Support

### Getting Help
1. Check logs: `docker-compose logs`
2. Run health check: `curl http://localhost:5000/api/health`
3. Test deployment: `python test_deployment.py`
4. Review configuration files
5. Check system resources

### Performance Metrics
- **Upload**: < 5 seconds for 1MB FASTA
- **Job Submission**: < 2 seconds
- **Status Updates**: Every 2 seconds
- **Memory Usage**: ~500MB base + job requirements
