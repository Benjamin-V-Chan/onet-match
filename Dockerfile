# docker build --build-context collector=../ONET-Data-Collector -t onet-match .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ONET_MATCH_BUNDLE=/app/bundle HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY --from=collector pyproject.toml README.md LICENSE /src/collector/
COPY --from=collector onet_data_collector /src/collector/onet_data_collector
RUN pip install /src/collector

COPY pyproject.toml README.md LICENSE /src/analysis/
COPY onet_analysis /src/analysis/onet_analysis
RUN pip install "/src/analysis[matching,serve]"

COPY models/bundle /app/bundle

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "onet_analysis.matching.api:app", "--host", "0.0.0.0", "--port", "8000"]
