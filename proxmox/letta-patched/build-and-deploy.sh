#!/bin/bash
# Build and deploy patched Letta image
# Usage: ./build-and-deploy.sh [--rebuild]

set -e

REMOTE_HOST="straughter@192.168.1.143"
IMAGE_NAME="letta-patched"
CONTAINER_NAME="letta-server"

echo "=========================================="
echo "Letta Patched Image Build & Deploy"
echo "=========================================="

# Copy Dockerfile to remote
echo "[1/5] Copying Dockerfile to remote..."
ssh $REMOTE_HOST "mkdir -p ~/letta-patched"
scp "$(dirname "$0")/Dockerfile" $REMOTE_HOST:~/letta-patched/

# Build image on remote
echo "[2/5] Building patched image on remote..."
ssh $REMOTE_HOST "cd ~/letta-patched && docker build -t $IMAGE_NAME ."

# Check if rebuild flag passed
if [[ "$1" == "--rebuild" ]]; then
    echo "[3/5] Stopping and removing existing container..."
    ssh $REMOTE_HOST "docker stop $CONTAINER_NAME 2>/dev/null || true"
    ssh $REMOTE_HOST "docker rm $CONTAINER_NAME 2>/dev/null || true"

    echo "[4/5] Starting new container..."
    # Get the original container's environment and ports
    ssh $REMOTE_HOST "docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        -p 8283:8283 \
        -e LETTA_PG_URI=\${LETTA_PG_URI:-postgresql+pg8000://letta:letta@host.docker.internal:5432/letta} \
        --add-host=host.docker.internal:host-gateway \
        $IMAGE_NAME"
else
    echo "[3/5] Restarting existing container with new image..."
    # For existing container, we need to recreate it
    echo "NOTE: To use the new image, run with --rebuild flag"
    echo "      ./build-and-deploy.sh --rebuild"
fi

# Verify
echo "[5/5] Verifying patches..."
sleep 5
ssh $REMOTE_HOST "docker exec $CONTAINER_NAME grep -q 'claude-max-router' /app/letta/schemas/llm_config.py && echo '✓ llm_config.py patch verified' || echo '✗ llm_config.py patch missing'"
ssh $REMOTE_HOST "docker exec $CONTAINER_NAME grep -q 'claude-max-router' /app/letta/schemas/model.py && echo '✓ model.py patch verified' || echo '✗ model.py patch missing'"

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "If patches are missing, run:"
echo "  $0 --rebuild"
