from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from analyzer import AudioAnalyzer
from translator import SunoTranslator
import shutil
import os

app = FastAPI(
    title="Music Prompt Generator API",
    description="分析音訊檔案並生成 SUNO 風格的 Prompt",
    version="1.0.0"
)

# 允許跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = AudioAnalyzer()
translator = SunoTranslator()

# 確保暫存目錄存在
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {
        "message": "Music Prompt Generator API",
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /api/analyze - 上傳音訊檔案進行分析"
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
    allowed_types = [
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave",
        "audio/x-wav", "audio/flac", "audio/ogg", "audio/x-flac"
    ]

    if file.content_type and file.content_type not in allowed_types:
        # 有些瀏覽器可能不會正確設定 content_type，所以也檢查副檔名
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.mp3', '.wav', '.flac', '.ogg']:
            raise HTTPException(
                status_code=400,
                detail=f"不支援的檔案格式: {file.content_type or ext}"
            )

    temp_path = os.path.join(TEMP_DIR, file.filename)

    try:
        # 儲存上傳的檔案
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 分析音訊
        raw_features = analyzer.analyze(temp_path)

        if not raw_features:
            raise HTTPException(
                status_code=400,
                detail="音訊分析失敗，請確認檔案是否為有效的音訊格式"
            )

        # 生成 Prompt
        result = translator.generate_prompt(raw_features)

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"伺服器錯誤: {str(e)}"
        )
    finally:
        # 清理暫存檔案
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
