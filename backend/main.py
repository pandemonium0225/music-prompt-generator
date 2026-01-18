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
CLAP_ENABLED = os.getenv("CLAP_ENABLED", "true").lower() == "true"

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
    logger.info(f"Starting Music Prompt Generator API v2.0 in {ENVIRONMENT} mode")
    logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")
    logger.info(f"CLAP enabled: {CLAP_ENABLED}")

    # 延遲載入重量級模組 (librosa 載入較慢)
    from analyzer import AudioAnalyzer
    from translator import SunoTranslator

    app.state.analyzer = AudioAnalyzer()
    app.state.translator = SunoTranslator()

    # 載入 CLAP 分析器 (如果啟用)
    app.state.clap_analyzer = None
    if CLAP_ENABLED:
        try:
            from clap_analyzer import get_clap_analyzer
            app.state.clap_analyzer = get_clap_analyzer()
            if app.state.clap_analyzer:
                logger.info("CLAP analyzer initialized (model will load on first use)")
        except Exception as e:
            logger.warning(f"Failed to initialize CLAP analyzer: {e}")
            logger.warning("Falling back to Librosa-only analysis")

    yield

    # 關閉時
    logger.info("Shutting down Music Prompt Generator API")


app = FastAPI(
    title="Music Prompt Generator API",
    description="分析音訊檔案並生成 SUNO 風格的 Prompt (支援 CLAP)",
    version="2.0.0",
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
        "version": "2.0.0",
        "environment": ENVIRONMENT,
        "clap_enabled": CLAP_ENABLED,
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

    分析方式:
    - Librosa: 提取 BPM, Key, Energy 等數值特徵
    - CLAP (可選): 使用 AI 模型進行語義風格分析
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

        # 1. Librosa 分析 (數值特徵)
        raw_features = app.state.analyzer.analyze(temp_path)

        if not raw_features:
            raise HTTPException(
                status_code=400,
                detail="音訊分析失敗，請確認檔案是否為有效的音訊格式"
            )

        # 2. CLAP 分析 (語義特徵) - 如果啟用
        clap_result = None
        clap_tags = []

        if app.state.clap_analyzer:
            try:
                logger.info("Running CLAP analysis...")
                clap_result = app.state.clap_analyzer.analyze(temp_path)

                if clap_result:
                    clap_tags = app.state.clap_analyzer.get_formatted_tags(clap_result)
                    logger.info(f"CLAP tags: {clap_tags}")
            except Exception as e:
                logger.warning(f"CLAP analysis failed: {e}")
                # CLAP 失敗不影響主要流程

        # 3. 生成 Prompt (結合 Librosa + CLAP)
        result = app.state.translator.generate_prompt(
            features=raw_features,
            clap_tags=clap_tags if clap_tags else None
        )

        # 加入 CLAP 詳細結果 (如果有)
        if clap_result:
            result["clap_analysis"] = {
                "tags": clap_result.get("tags", [])[:10],
                "top_by_category": clap_result.get("top_tags_by_category", {})
            }

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
        "version": "2.0.0",
        "environment": ENVIRONMENT,
        "clap_enabled": CLAP_ENABLED
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=(ENVIRONMENT == "development")
    )
