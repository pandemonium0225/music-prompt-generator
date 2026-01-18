# syntax=docker/dockerfile:1

# =============================================================================
# Music Prompt Generator - Frontend Dockerfile
# =============================================================================
# 多階段建構：Build → Production
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - 建構 React 應用
# -----------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# 複製 package 檔案
COPY package.json package-lock.json* ./

# 安裝依賴
RUN npm ci

# 複製源碼
COPY . .

# 建構參數 (可在建構時覆蓋)
ARG VITE_API_URL=/api

# 設定環境變數供 Vite 使用
ENV VITE_API_URL=${VITE_API_URL}

# 建構生產版本
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Production - Nginx 靜態檔案伺服器
# -----------------------------------------------------------------------------
FROM nginx:alpine AS production

# 複製自訂 nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 從 builder 複製建構產物
COPY --from=builder /app/dist /usr/share/nginx/html

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

# 暴露端口 (Cloud Run 預設使用 8080)
EXPOSE 8080

# 啟動 nginx
CMD ["nginx", "-g", "daemon off;"]
