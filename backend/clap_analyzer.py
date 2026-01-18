"""
CLAP (Contrastive Language-Audio Pretraining) 分析器

使用 LAION CLAP 模型進行 Zero-shot 音樂風格分類，
透過音訊-文字相似度計算，找出最匹配的音樂風格標籤。

參考: https://huggingface.co/laion/larger_clap_music
"""

import logging
import os
from typing import Optional

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

logger = logging.getLogger(__name__)

# 是否啟用 CLAP (可透過環境變數控制)
CLAP_ENABLED = os.getenv("CLAP_ENABLED", "true").lower() == "true"
CLAP_MODEL_NAME = os.getenv("CLAP_MODEL", "laion/larger_clap_music")
CLAP_TOP_K = int(os.getenv("CLAP_TOP_K", "8"))

# 類別配置：權重、最大數量、閾值
CATEGORY_CONFIG = {
    "genre": {"weight": 1.0, "max": 2, "threshold": 0.008},
    "mood": {"weight": 0.9, "max": 2, "threshold": 0.007},
    "instruments": {"weight": 1.1, "max": 2, "threshold": 0.008},
    "vocals": {"weight": 0.8, "max": 1, "threshold": 0.008},
    "production": {"weight": 0.7, "max": 1, "threshold": 0.007},
    "rhythm": {"weight": 0.9, "max": 1, "threshold": 0.008},
    "era": {"weight": 0.5, "max": 1, "threshold": 0.009},
    "atmosphere": {"weight": 0.6, "max": 1, "threshold": 0.007},
}

# CLAP 信心度閾值：當最高分低於此值時，認為 CLAP 結果不可靠
CLAP_CONFIDENCE_THRESHOLD = float(os.getenv("CLAP_CONFIDENCE_THRESHOLD", "0.015"))


class CLAPAnalyzer:
    """
    CLAP 音樂風格分析器

    使用 Zero-shot 分類方式，從預定義的標籤庫中找出最匹配的風格描述。
    """

    # 音樂風格標籤庫 (針對 SUNO 優化)
    MUSIC_TAGS = {
        # 曲風 (Genre)
        "genre": [
            "pop music", "rock music", "electronic music", "hip hop",
            "jazz", "classical music", "R&B", "country music",
            "folk music", "blues", "reggae", "soul music",
            "metal", "punk rock", "indie rock", "alternative rock",
            "house music", "techno", "trance", "dubstep",
            "ambient music", "lo-fi hip hop", "trap music", "drill",
            "disco", "funk", "gospel", "latin music",
            "K-pop", "J-pop", "Afrobeat", "world music",
        ],

        # 情緒 (Mood)
        "mood": [
            "happy and uplifting", "sad and melancholic", "energetic and exciting",
            "calm and relaxing", "dark and intense", "romantic and emotional",
            "angry and aggressive", "peaceful and serene", "mysterious and suspenseful",
            "nostalgic and dreamy", "epic and cinematic", "playful and fun",
            "anxious and tense", "hopeful and inspiring", "lonely and isolated",
        ],

        # 樂器 (Instruments)
        "instruments": [
            "piano", "acoustic guitar", "electric guitar", "synthesizer",
            "drums", "bass guitar", "violin", "orchestra",
            "saxophone", "trumpet", "flute", "cello",
            "808 drums", "drum machine", "synth bass", "organ",
            "ukulele", "harmonica", "banjo", "harp",
        ],

        # 人聲 (Vocals)
        "vocals": [
            "male vocals", "female vocals", "no vocals instrumental",
            "choir vocals", "rap vocals", "auto-tuned vocals",
            "whispered vocals", "powerful vocals", "soft vocals",
            "duet vocals", "backing vocals", "spoken word",
        ],

        # 製作風格 (Production)
        "production": [
            "lo-fi production", "high fidelity", "vintage analog",
            "modern digital", "acoustic recording", "heavy bass",
            "clean mix", "distorted", "reverb heavy", "dry mix",
            "compressed", "dynamic range", "stereo wide", "mono",
        ],

        # 時代風格 (Era)
        "era": [
            "80s style", "90s style", "2000s style", "2010s style",
            "modern contemporary", "retro vintage", "futuristic",
            "classic timeless", "old school", "new wave",
        ],

        # 節奏特徵 (Rhythm)
        "rhythm": [
            "four on the floor beat", "syncopated rhythm", "swing rhythm",
            "steady beat", "complex polyrhythm", "simple rhythm",
            "groovy", "driving beat", "laid back tempo", "breakbeat",
        ],

        # 氛圍 (Atmosphere)
        "atmosphere": [
            "atmospheric and spacious", "intimate and close",
            "outdoor nature sounds", "urban city vibes",
            "club dance floor", "concert live performance",
            "bedroom studio", "professional studio",
        ],
    }

    def __init__(self, model_name: str = CLAP_MODEL_NAME, device: Optional[str] = None):
        """
        初始化 CLAP 分析器

        Args:
            model_name: HuggingFace 模型名稱
            device: 運算裝置 ('cuda', 'cpu', 或 None 自動選擇)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._all_tags = None
        self._tag_embeddings = None

        logger.info(f"CLAP Analyzer initialized (model: {model_name}, device: {self.device})")

    def _load_model(self):
        """延遲載入模型 (首次使用時才載入)"""
        if self.model is not None:
            return

        logger.info(f"Loading CLAP model: {self.model_name}")

        try:
            self.processor = ClapProcessor.from_pretrained(self.model_name)
            self.model = ClapModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

            # 預先計算所有標籤的 embeddings
            self._prepare_tag_embeddings()

            logger.info("CLAP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLAP model: {e}")
            raise

    def _prepare_tag_embeddings(self):
        """預先計算所有標籤的文字 embeddings"""
        # 展平所有標籤
        self._all_tags = []
        self._tag_categories = {}

        for category, tags in self.MUSIC_TAGS.items():
            for tag in tags:
                self._all_tags.append(tag)
                self._tag_categories[tag] = category

        logger.debug(f"Preparing embeddings for {len(self._all_tags)} tags")

        # 計算文字 embeddings
        with torch.no_grad():
            text_inputs = self.processor(
                text=self._all_tags,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            self._tag_embeddings = self.model.get_text_features(**text_inputs)
            self._tag_embeddings = self._tag_embeddings / self._tag_embeddings.norm(dim=-1, keepdim=True)

        logger.debug("Tag embeddings prepared")

    def analyze(self, file_path: str, top_k: int = CLAP_TOP_K) -> Optional[dict]:
        """
        分析音訊檔案，返回最匹配的音樂風格標籤

        Args:
            file_path: 音訊檔案路徑
            top_k: 返回前 k 個最匹配的標籤

        Returns:
            dict: 包含 tags, categories, scores 的分析結果
        """
        if not CLAP_ENABLED:
            logger.info("CLAP is disabled, skipping analysis")
            return None

        try:
            # 延遲載入模型
            self._load_model()

            # 載入音訊 (CLAP 使用 48kHz)
            logger.debug(f"Loading audio: {file_path}")
            audio, sr = librosa.load(file_path, sr=48000, duration=30)

            if len(audio) == 0:
                logger.error("Empty audio file")
                return None

            # 計算音訊 embedding
            with torch.no_grad():
                audio_inputs = self.processor(
                    audios=audio,
                    sampling_rate=48000,
                    return_tensors="pt"
                ).to(self.device)

                audio_embedding = self.model.get_audio_features(**audio_inputs)
                audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)

            # 計算相似度
            similarities = (audio_embedding @ self._tag_embeddings.T).squeeze(0)
            similarities = similarities.cpu().numpy()

            # 取得 top-k 結果
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = {
                "tags": [],
                "categories": {},
                "scores": {},
                "top_tags_by_category": {},
            }

            # 整理結果
            category_scores = {cat: [] for cat in self.MUSIC_TAGS.keys()}

            for idx in top_indices:
                tag = self._all_tags[idx]
                score = float(similarities[idx])
                category = self._tag_categories[tag]

                results["tags"].append(tag)
                results["scores"][tag] = round(score, 4)
                results["categories"][tag] = category
                category_scores[category].append((tag, score))

            # 每個類別取最高分的標籤
            for category, scores in category_scores.items():
                if scores:
                    best_tag, best_score = max(scores, key=lambda x: x[1])
                    results["top_tags_by_category"][category] = {
                        "tag": best_tag,
                        "score": round(best_score, 4)
                    }

            # 補充每個類別的最佳標籤 (即使不在 top_k 中)
            for category, tags in self.MUSIC_TAGS.items():
                if category not in results["top_tags_by_category"]:
                    # 找出該類別中分數最高的標籤
                    tag_indices = [self._all_tags.index(t) for t in tags]
                    tag_scores = [(tags[i], similarities[tag_indices[i]]) for i in range(len(tags))]
                    best_tag, best_score = max(tag_scores, key=lambda x: x[1])
                    results["top_tags_by_category"][category] = {
                        "tag": best_tag,
                        "score": round(float(best_score), 4)
                    }

            logger.info(f"CLAP analysis complete: {results['tags'][:5]}...")
            return results

        except Exception as e:
            logger.error(f"CLAP analysis error: {e}", exc_info=True)
            return None

    def get_formatted_tags(self, analysis_result: dict, max_tags: int = 10) -> dict:
        """
        從分析結果中提取格式化的標籤列表（按類別組織）

        使用 CATEGORY_CONFIG 的權重和閾值進行選取:
        - 每個類別有獨立的閾值（基於實際 CLAP 分數範圍 0.01-0.03）
        - 按加權分數排序選取
        - 每個類別最多選取 max 個標籤
        - 當最高分低於 CLAP_CONFIDENCE_THRESHOLD 時，返回空結果（不可靠）

        Returns:
            dict: {
                "by_category": {"genre": [...], "mood": [...], ...},
                "all_tags": [...],
                "weighted_scores": {...},
                "confidence": float,  # 最高分數
                "is_reliable": bool   # 是否可靠
            }
        """
        empty_result = {
            "by_category": {},
            "all_tags": [],
            "weighted_scores": {},
            "confidence": 0.0,
            "is_reliable": False
        }

        if not analysis_result:
            return empty_result

        # 檢查 CLAP 信心度
        all_scores = analysis_result.get("scores", {})
        if not all_scores:
            return empty_result

        max_score = max(all_scores.values()) if all_scores else 0.0
        is_reliable = max_score >= CLAP_CONFIDENCE_THRESHOLD

        if not is_reliable:
            logger.info(
                f"CLAP confidence too low ({max_score:.4f} < {CLAP_CONFIDENCE_THRESHOLD}), "
                "ignoring CLAP results"
            )
            return {
                "by_category": {},
                "all_tags": [],
                "weighted_scores": {},
                "confidence": max_score,
                "is_reliable": False
            }

        # 收集每個類別的候選標籤
        category_candidates = {cat: [] for cat in CATEGORY_CONFIG.keys()}

        # 從所有標籤中收集
        all_scores = analysis_result.get("scores", {})
        all_categories = analysis_result.get("categories", {})

        for tag, score in all_scores.items():
            category = all_categories.get(tag, "")
            if category in CATEGORY_CONFIG:
                config = CATEGORY_CONFIG[category]
                # 計算加權分數
                weighted_score = score * config["weight"]
                category_candidates[category].append({
                    "tag": tag,
                    "score": score,
                    "weighted_score": weighted_score
                })

        # 補充 top_tags_by_category 中的標籤（可能不在 top-k 中）
        for category, tag_info in analysis_result.get("top_tags_by_category", {}).items():
            if category in CATEGORY_CONFIG:
                tag = tag_info["tag"]
                score = tag_info["score"]
                # 檢查是否已存在
                existing_tags = [c["tag"] for c in category_candidates[category]]
                if tag not in existing_tags:
                    config = CATEGORY_CONFIG[category]
                    weighted_score = score * config["weight"]
                    category_candidates[category].append({
                        "tag": tag,
                        "score": score,
                        "weighted_score": weighted_score
                    })

        # 按類別選取標籤
        by_category = {}
        all_tags = []
        weighted_scores = {}

        for category, config in CATEGORY_CONFIG.items():
            candidates = category_candidates.get(category, [])
            if not candidates:
                continue

            # 按加權分數排序
            candidates.sort(key=lambda x: x["weighted_score"], reverse=True)

            # 選取達到閾值的標籤
            selected = []
            for c in candidates:
                if len(selected) >= config["max"]:
                    break
                if c["score"] >= config["threshold"]:
                    # 清理標籤
                    clean_tag = self._clean_tag(c["tag"])
                    if clean_tag and clean_tag not in selected:
                        selected.append(clean_tag)
                        weighted_scores[clean_tag] = round(c["weighted_score"], 4)

            if selected:
                by_category[category] = selected
                all_tags.extend(selected)

        return {
            "by_category": by_category,
            "all_tags": all_tags[:max_tags],
            "weighted_scores": weighted_scores,
            "confidence": max_score,
            "is_reliable": True
        }

    def _clean_tag(self, tag: str) -> str:
        """清理標籤，移除冗餘後綴"""
        if not tag:
            return ""

        # 移除常見後綴
        suffixes = [" music", " style", " recording", " production"]
        result = tag
        for suffix in suffixes:
            if result.endswith(suffix):
                result = result[:-len(suffix)]

        return result.strip()


# 單例模式 (避免重複載入模型)
_clap_instance: Optional[CLAPAnalyzer] = None


def get_clap_analyzer() -> Optional[CLAPAnalyzer]:
    """取得 CLAP 分析器單例"""
    global _clap_instance

    if not CLAP_ENABLED:
        return None

    if _clap_instance is None:
        _clap_instance = CLAPAnalyzer()

    return _clap_instance
