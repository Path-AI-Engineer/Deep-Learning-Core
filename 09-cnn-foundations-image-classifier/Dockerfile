FROM node:24.13-alpine AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12.8-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    CNN_BUNDLE_PATH=/app/artifacts/models/cnn/v1.0.0 \
    FASHION_MNIST_ROOT=/app/data/raw
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --uid 10001 app
COPY requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt
COPY src/ src/
COPY backend/ backend/
COPY configs/ configs/
COPY artifacts/models/cnn/v1.0.0/ artifacts/models/cnn/v1.0.0/
COPY data/raw/FashionMNIST/ data/raw/FashionMNIST/
COPY --from=frontend /web/dist frontend/dist/
RUN test -f artifacts/models/cnn/v1.0.0/model_state.pt \
    && test -f artifacts/models/cnn/v1.0.0/manifest.json \
    && chown -R app:app /app
USER app
ENV PYTHONPATH=/app/src:/app/backend
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
