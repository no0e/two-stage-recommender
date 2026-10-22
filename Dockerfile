# The API image: ONNX Runtime and numpy, no PyTorch.
#
#   docker build -t recsys-api .
#   docker run -p 8000:8000 -v $PWD/serving_artifacts/Video_Games:/app/artifacts recsys-api
#
# Artifacts are mounted, or baked in with --build-arg ARTIFACTS=serving_artifacts/<category>
# (which is what deploy/cloudrun.sh does, since Cloud Run has no volume to mount).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts PORT=8000 WORKERS=1

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY recsys/__init__.py recsys/data.py recsys/
COPY recsys/serving recsys/serving
ARG ARTIFACTS=serving_artifacts
COPY ${ARTIFACTS} /app/artifacts

RUN useradd --create-home api && chown -R api /app
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c \
    "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn --factory recsys.serving.app:create_app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
