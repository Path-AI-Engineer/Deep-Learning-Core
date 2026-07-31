FROM node:24-alpine AS frontend
WORKDIR /workspace/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/backend \
    PORT=8080 \
    TRANSFORMER_BUNDLE_PATH=/app/artifacts/models/transformer/v1.0.0-reference \
    TRANSFORMER_SAMPLE_PATH=/app/data/samples/demo_catalog.json
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY backend ./backend
COPY artifacts ./artifacts
COPY data/samples ./data/samples
COPY reports/predictions ./reports/predictions
COPY paper ./paper
COPY --from=frontend /workspace/frontend/dist ./frontend/dist
EXPOSE 8080
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
