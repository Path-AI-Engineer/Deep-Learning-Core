FROM python:3.12.7-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --uid 10001 app

COPY requirements.txt ./
RUN python -m pip install --upgrade "pip==24.2" \
    && python -m pip install --requirement requirements.txt

COPY src ./src
COPY frontend ./frontend
COPY configs ./configs
COPY contracts ./contracts

RUN chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/_stcore/health', timeout=3)"

CMD ["sh", "-c", "python -m streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true --browser.gatherUsageStats=false"]
