# Dockerfile for Letta with claude-max-router support
# This image patches Letta to accept custom model endpoint types

ARG LETTA_VERSION=latest
FROM letta/letta:${LETTA_VERSION}

LABEL maintainer="Letta MAS Admin"
LABEL description="Letta with claude-max-router endpoint support"

# Patch llm_config.py to add claude-max-router to allowed endpoint types
RUN sed -i 's/"xai",/"xai",\n    "claude-max-router",/' /app/letta/schemas/llm_config.py || \
    echo "Warning: Could not patch llm_config.py"

# Patch model.py to add claude-max-router to allowed endpoint types
RUN sed -i 's/"xai",/"xai",\n    "claude-max-router",/' /app/letta/schemas/model.py || \
    echo "Warning: Could not patch model.py"

# Verify patches were applied
RUN grep -q "claude-max-router" /app/letta/schemas/llm_config.py && \
    echo "✓ llm_config.py patched successfully" || \
    echo "✗ llm_config.py patch failed"

RUN grep -q "claude-max-router" /app/letta/schemas/model.py && \
    echo "✓ model.py patched successfully" || \
    echo "✗ model.py patch failed"

# Keep the original entrypoint
ENTRYPOINT ["./letta/server/startup.sh"]
