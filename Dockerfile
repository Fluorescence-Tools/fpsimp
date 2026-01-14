# syntax=docker/dockerfile:1
FROM continuumio/miniconda3:latest

# Add conda to PATH (already set in base image, but being explicit)
ENV PATH="/opt/conda/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy environment file and create conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment by default
ENV CONDA_DEFAULT_ENV=fpsimp
ENV PATH="/opt/conda/envs/fpsimp/bin:$PATH"

# Create necessary directories (code mounted via volume)
RUN mkdir -p uploads results templates static

# Expose port
EXPOSE 5000

# Default command (can be overridden in docker-compose)
CMD ["conda", "run", "-n", "fpsimp", "python", "src/app.py"]
