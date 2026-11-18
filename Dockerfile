# FinAccess bank-use API - multi-stage build (course ch. 10 conventions)

# ---- Stage 1: install dependencies into a virtual environment -------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first, so this layer is cached until requirements change.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# ---- Stage 2: slim runtime image ------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="finaccess-bank-api" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="Predicts traditional bank use (FinAccess 2024); DSA 8401 capstone"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/model_package.joblib \
    LOG_LEVEL=info

# Non-root user.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Application code, then the model package baked into the image.
COPY api/ api/
COPY models/model_package.joblib models/model_package.joblib

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
