# Music Prompt Generator

一個音訊分析工具，可以解析歌曲的音樂特徵（BPM、調性、能量等），並自動生成適用於 SUNO AI 的風格提示詞 (Prompt)。

## 功能特色

- 上傳 MP3/WAV/FLAC/OGG 音訊檔案
- 自動分析音樂特徵：
  - BPM (節奏速度)
  - Key & Mode (調性與大小調)
  - Energy (能量強度)
  - Brightness (音色明亮度)
  - Rhythm Strength (節奏強度)
- 智慧生成 SUNO 風格標籤
- 一鍵複製 Prompt

## 專案結構

```
music-prompt-generator/
├── backend/                  # Python FastAPI 後端
│   ├── main.py              # API 入口
│   ├── analyzer.py          # 音訊分析 (Librosa)
│   ├── translator.py        # 特徵轉 Prompt
│   ├── pyproject.toml       # Python 依賴 (uv)
│   ├── Dockerfile           # Docker 映像配置
│   ├── .dockerignore
│   └── .env.example         # 環境變數範本
├── frontend/                 # React + Vite 前端
│   ├── src/
│   │   ├── App.jsx          # 主要介面
│   │   ├── main.jsx         # 入口點
│   │   └── index.css        # 樣式
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile           # Docker 映像配置
│   ├── nginx.conf           # Nginx 配置
│   ├── .dockerignore
│   └── .env.example         # 環境變數範本
├── docker-compose.yml        # 本地 Docker 測試
├── cloudbuild.yaml           # GCP Cloud Build 配置
└── README.md
```

---

## 快速開始

### 環境需求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) - Python 套件管理器
- Node.js 18+
- npm

### 方式一：本地開發 (推薦)

**1. 啟動後端**

```bash
cd backend
uv run python main.py
```

後端將在 http://localhost:8000 運行

**2. 啟動前端**

```bash
cd frontend
npm install
npm run dev
```

前端將在 http://localhost:5173 運行

### 方式二：Docker Compose

```bash
# 建構並啟動所有服務
docker compose up --build

# 背景執行
docker compose up -d --build

# 查看日誌
docker compose logs -f

# 停止服務
docker compose down
```

- 前端: http://localhost:3000
- 後端: http://localhost:8000

---

## 環境變數

### Backend

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `PORT` | `8000` | 伺服器端口 |
| `ENVIRONMENT` | `development` | 執行環境 (development/production) |
| `LOG_LEVEL` | `INFO` | 日誌等級 (DEBUG/INFO/WARNING/ERROR) |
| `ALLOWED_ORIGINS` | `*` | CORS 允許的來源 (逗號分隔) |

### Frontend

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `VITE_API_URL` | `/api` | API 伺服器網址 |

---

## GCP 部署指南

### 前置條件

1. 安裝 [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. 建立 GCP 專案並啟用以下 API：
   - Cloud Run API
   - Cloud Build API
   - Container Registry API

```bash
# 登入 GCP
gcloud auth login

# 設定專案
gcloud config set project YOUR_PROJECT_ID

# 啟用 API
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com
```

### 部署方式

#### 方式一：使用 Cloud Build (CI/CD)

```bash
# 從專案根目錄執行
gcloud builds submit --config cloudbuild.yaml
```

這會自動：
1. 建構 Backend 和 Frontend Docker 映像
2. 推送到 Container Registry
3. 部署到 Cloud Run

#### 方式二：手動部署

**部署 Backend:**

```bash
cd backend

# 建構映像
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/music-prompt-backend

# 部署到 Cloud Run
gcloud run deploy music-prompt-backend \
  --image gcr.io/YOUR_PROJECT_ID/music-prompt-backend \
  --region asia-east1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO"
```

**部署 Frontend:**

```bash
cd frontend

# 建構映像 (替換 BACKEND_URL 為實際的後端網址)
gcloud builds submit \
  --tag gcr.io/YOUR_PROJECT_ID/music-prompt-frontend \
  --build-arg VITE_API_URL=https://YOUR_BACKEND_URL/api

# 部署到 Cloud Run
gcloud run deploy music-prompt-frontend \
  --image gcr.io/YOUR_PROJECT_ID/music-prompt-frontend \
  --region asia-east1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 256Mi
```

### 部署後設定

部署完成後，需要更新 Backend 的 CORS 設定：

```bash
gcloud run services update music-prompt-backend \
  --region asia-east1 \
  --set-env-vars "ALLOWED_ORIGINS=https://YOUR_FRONTEND_URL"
```

---

## API 端點

### GET /

API 資訊

### POST /api/analyze

上傳音訊檔案進行分析

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - 音訊檔案

**Response:**
```json
{
  "success": true,
  "data": {
    "prompt": "upbeat, energetic, electronic, bright, pop, C major, 128 bpm",
    "tags": ["upbeat", "energetic", "electronic", "bright", "pop"],
    "key": "C major",
    "bpm": 128,
    "features": {
      "bpm": 128,
      "energy": 0.1523,
      "brightness": 2845.32,
      "rhythm": 1.234,
      "key": "C",
      "mode": "major"
    }
  }
}
```

### GET /api/health

健康檢查端點

---

## 技術棧

### 後端
- FastAPI - Web 框架
- Librosa - 音訊分析
- NumPy / SciPy - 數學運算
- Gunicorn + Uvicorn - 生產環境伺服器

### 前端
- React 18
- Vite - 建構工具
- Tailwind CSS - 樣式
- Axios - HTTP 請求
- Lucide React - 圖示

### 部署
- Docker - 容器化
- Nginx - 靜態檔案伺服器
- GCP Cloud Run - Serverless 容器平台
- GCP Cloud Build - CI/CD

---

## 授權

MIT License
