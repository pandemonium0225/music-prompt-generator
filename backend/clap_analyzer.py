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

    def get_formatted_tags(self, analysis_result: dict, max_tags: int = 6) -> list:
        """
        從分析結果中提取格式化的標籤列表

        優先從不同類別中各選一個，確保多樣性
        """
        if not analysis_result:
            return []

        selected = []
        used_categories = set()

        # 優先類別順序
        priority_categories = ["genre", "mood", "instruments", "vocals", "production", "era"]

        # 先從每個優先類別取一個
        for category in priority_categories:
            if len(selected) >= max_tags:
                break
            if category in analysis_result.get("top_tags_by_category", {}):
                tag_info = analysis_result["top_tags_by_category"][category]
                if tag_info["score"] > 0.1:  # 只取有足夠信心的標籤
                    # 簡化標籤 (移除 " music" 等後綴)
                    tag = tag_info["tag"]
                    tag = tag.replace(" music", "").replace(" style", "")
                    if tag not in selected:
                        selected.append(tag)
                        used_categories.add(category)

        # 如果還有空間，從 top tags 補充
        for tag in analysis_result.get("tags", []):
            if len(selected) >= max_tags:
                break
            category = analysis_result["categories"].get(tag, "")
            if category not in used_categories:
                tag_clean = tag.replace(" music", "").replace(" style", "")
                if tag_clean not in selected:
                    selected.append(tag_clean)
                    used_categories.add(category)

        return selected


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
