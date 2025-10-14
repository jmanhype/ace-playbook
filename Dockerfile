# ACE Playbook - Production Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ./
COPY ace/ ./ace/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Install dependencies
RUN uv pip install --system -e ".[dev]"

# Create volume mount point for database
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:////data/ace_playbook.db

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "from ace.utils.database import get_engine; get_engine()" || exit 1

# Run database migrations on startup
CMD alembic upgrade head && python -m ace.cli

# Expose metrics endpoint (if running web server)
EXPOSE 8000
