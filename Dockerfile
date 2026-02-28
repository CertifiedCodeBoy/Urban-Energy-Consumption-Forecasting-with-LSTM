# ---------------------------------------------------------------------------
# Stage 1: build dependencies
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build tools (needed for some compiled packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY config.py   .
COPY serve.py    .
COPY train.py    .
COPY src/        src/

# Create expected data directories (volumes can be mounted over these)
RUN mkdir -p data/raw data/processed models

# Non-root user for safety
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check — waits 15s for the model to load, then checks every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
