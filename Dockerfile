# Multi-stage Dockerfile for Fraud Detection Pipeline
# Stage 1: Build environment
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source files needed for build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies and build wheel
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels .

# Stage 2: Runtime environment
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy source code
COPY src/ ./src/
COPY sql/ ./sql/
COPY dags/ ./dags/

# Create directories
RUN mkdir -p /app/data /app/models /app/results /app/reports

# Set environment variables
ENV PYTHONPATH=/app/src
ENV FRAUD_DATA_DIR=/app/data
ENV FRAUD_MODELS_DIR=/app/models
ENV FRAUD_RESULTS_DIR=/app/results
ENV FRAUD_REPORTS_DIR=/app/reports

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from fraud_detection.utils.config import get_settings; get_settings()"

# Default command (can be overridden)
CMD ["python", "-m", "fraud_detection.cli", "--help"]
