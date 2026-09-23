FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9 AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt constraints.txt requirements-ai.txt ./
ARG INSTALL_AI=false
RUN python -m pip install -r requirements.txt && \
    if [ "$INSTALL_AI" = "true" ]; then python -m pip install -r requirements-ai.txt; fi

FROM dependencies AS runtime
COPY . .

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.serverAddress=localhost"]

FROM dependencies AS test
RUN apt-get update && apt-get install -y --no-install-recommends chromium && rm -rf /var/lib/apt/lists/*
COPY requirements-dev.txt ./
RUN python -m pip install -r requirements-dev.txt
COPY . .
CMD ["python", "-m", "pytest", "-q"]
