# One image, two entrypoints (API and Streamlit UI) so both always ship the same model.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src/churn/__init__.py src/churn/__init__.py
RUN pip install ".[api,app]"

COPY src ./src
COPY api ./api
COPY app ./app
COPY data ./data
COPY models ./models
COPY reports ./reports
RUN pip install --no-deps -e .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fs http://localhost:8000/health || curl -fs http://localhost:8501/_stcore/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
