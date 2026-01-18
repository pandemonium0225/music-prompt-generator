# syntax=docker/dockerfile:1

# =============================================================================
# Music Prompt Generator - Backend Dockerfile
# =============================================================================
# 多階段建構：優化映像大小與建構快取
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - 安裝依賴
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# 安裝 uv (快速 Python 套件管理器)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 設定 uv 環境變數
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# 複製依賴定義檔
COPY pyproject.toml ./

# 安裝依賴到 .venv (不安裝專案本身)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

# -----------------------------------------------------------------------------
# Stage 2: Runtime - 最終運行環境
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# 安裝音訊處理系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 從 builder 複製虛擬環境
COPY --from=builder /app/.venv /app/.venv

# 設定 PATH 使用虛擬環境
ENV PATH="/app/.venv/bin:$PATH"

# 複製應用程式碼
COPY main.py analyzer.py translator.py ./

# 環境變數 (可被 Cloud Run 覆蓋)
ENV PORT=8000
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/health')" || exit 1

# 暴露端口
EXPOSE ${PORT}

# 啟動命令 - 使用 gunicorn + uvicorn worker (生產環境)
CMD ["sh", "-c", "gunicorn main:app --bind 0.0.0.0:${PORT} --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 120"]
