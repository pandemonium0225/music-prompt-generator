from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import tempfile
import logging

# 環境變數配置 (帶有預設值，確保向下相容)
PORT = int(os.getenv("PORT", 8000))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# 配置 logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    # 啟動時
    logger.info(f"Starting Music Prompt Generator API in {ENVIRONMENT} mode")
    logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")

    # 延遲載入重量級模組 (librosa 載入較慢)
    from analyzer import AudioAnalyzer
    from translator import SunoTranslator

    app.state.analyzer = AudioAnalyzer()
    app.state.translator = SunoTranslator()

    yield

    # 關閉時
    logger.info("Shutting down Music Prompt Generator API")


app = FastAPI(
    title="Music Prompt Generator API",
    description="分析音訊檔案並生成 SUNO 風格的 Prompt",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置 - 支援環境變數
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Music Prompt Generator API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /api/analyze - 上傳音訊檔案進行分析",
            "health": "GET /api/health - 健康檢查"
        }
    }


@app.post("/api/analyze")
async def analyze_song(file: UploadFile = File(...)):
    """
    分析上傳的音訊檔案，返回 SUNO 風格的 Prompt

    支援格式: MP3, WAV, FLAC, OGG
    建議檔案大小: < 10MB
    """
    # 驗證檔案類型
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg'}
    allowed_mimetypes = {
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave",
        "audio/x-wav", "audio/flac", "audio/ogg", "audio/x-flac"
    }

    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""

    if file.content_type and file.content_type not in allowed_mimetypes:
        if ext not in allowed_extensions:
            logger.warning(f"Rejected file with type: {file.content_type}, ext: {ext}")
            raise HTTPException(
                status_code=400,
                detail=f"不支援的檔案格式: {file.content_type or ext}"
            )

    # 使用系統暫存目錄 (Cloud Run 相容)
    temp_file = None
    temp_path = None
    try:
        # 建立暫存檔案
        suffix = ext if ext else ".tmp"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name

        # 寫入上傳的檔案內容
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        logger.info(f"Analyzing file: {file.filename} ({len(content)} bytes)")

        # 分析音訊
        raw_features = app.state.analyzer.analyze(temp_path)

        if not raw_features:
            raise HTTPException(
                status_code=400,
                detail="音訊分析失敗，請確認檔案是否為有效的音訊格式"
            )

        # 生成 Prompt
        result = app.state.translator.generate_prompt(raw_features)

        logger.info(f"Analysis complete: BPM={result['bpm']}, Key={result['key']}")

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"伺服器錯誤: {str(e)}"
        )
    finally:
        # 清理暫存檔案
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.get("/api/health")
async def health_check():
    """健康檢查端點 (供 Cloud Run / Load Balancer 使用)"""
    return {
        "status": "healthy",
        "environment": ENVIRONMENT
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=(ENVIRONMENT == "development")
    )
