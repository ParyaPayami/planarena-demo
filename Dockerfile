FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Render sets PORT; the free plan has 512 MB, so one worker.
CMD ["sh", "-c", "python -m uvicorn planarena.api.app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
