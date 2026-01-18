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
│   └── temp/                # 暫存目錄
├── frontend/                 # React + Vite 前端
│   ├── src/
│   │   ├── App.jsx          # 主要介面
│   │   ├── main.jsx         # 入口點
│   │   └── index.css        # 樣式
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 快速開始

### 環境需求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) - Python 套件管理器
- Node.js 18+
- npm 或 yarn

### 1. 啟動後端 (Python + uv)

```bash
cd backend

# 安裝依賴並啟動伺服器
uv run python main.py
```

或者分步執行：

```bash
cd backend

# 同步依賴
uv sync

# 啟動伺服器
uv run python main.py
```

後端將在 http://localhost:8000 運行

### 2. 啟動前端 (React)

```bash
cd frontend

# 安裝依賴
npm install

# 啟動開發伺服器
npm run dev
```

前端將在 http://localhost:5173 運行

## API 端點

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

## 技術棧

### 後端
- FastAPI - Web 框架
- Librosa - 音訊分析
- NumPy / SciPy - 數學運算

### 前端
- React 18
- Vite - 建構工具
- Tailwind CSS - 樣式
- Axios - HTTP 請求
- Lucide React - 圖示

## 授權

MIT License
