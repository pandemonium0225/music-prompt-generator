"""
SUNO Prompt 翻譯器

將音訊分析結果轉換為 SUNO AI 優化的 Prompt 格式

特性:
- LLM 自然語言生成：使用 GPT 生成流暢的音樂描述
- 語義衝突檢測：避免矛盾標籤（如 "ambient" + "powerful"）
- 多特徵組合分析：取代單一閾值判斷
- 結構化 Prompt 備援：當 LLM 不可用時使用標籤組合
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# 語義衝突配置
# ============================================================================

SEMANTIC_CONFLICTS = {
    "energy": {
        "high": ["upbeat", "energetic", "powerful", "intense", "driving", "dynamic", "anthemic", "explosive"],
        "low": ["ambient", "minimalist", "soft", "gentle", "relaxed", "calm", "ethereal", "peaceful"]
    },
    "tempo": {
        "fast": ["fast-paced", "frenetic", "high energy", "driving beat", "upbeat"],
        "slow": ["slow burn", "downtempo", "ballad", "contemplative", "laid back"]
    },
    "production": {
        "clean": ["polished", "crisp", "high fidelity", "clean mix", "modern"],
        "raw": ["lo-fi", "vintage", "distorted", "compressed", "warm", "analog"]
    },
    "mood": {
        "positive": ["uplifting", "happy", "bright mood", "hopeful", "playful"],
        "negative": ["melancholic", "dark", "sad", "lonely", "anxious"]
    }
}

# 語義相似詞組（用於去重）
SEMANTIC_SYNONYMS = {
    "energetic": ["upbeat", "high energy", "dynamic"],
    "ambient": ["ethereal", "atmospheric", "spacious"],
    "electronic": ["synthesizer", "synth", "digital"],
    "drums": ["drum machine", "808 drums", "percussive"],
    "soft": ["gentle", "delicate", "subtle"],
    "dark": ["deep", "moody", "shadowy"],
    "bright": ["crisp", "shimmery", "sparkling"],
}


class SunoTranslator:
    """
    將音訊分析結果轉換為 SUNO 風格的 Prompt

    支援兩種分析來源:
    - Librosa: 數值特徵 (BPM, Energy, Key...) + 正規化特徵
    - CLAP: 語義標籤（按類別組織）
    """

    def generate_prompt(
        self,
        features: dict,
        clap_result: Optional[dict] = None,
        normalized_features: Optional[dict] = None,
        use_llm: bool = True,
        llm_generator=None
    ) -> dict:
        """
        生成 SUNO 優化的 Prompt

        Args:
            features: 原始 Librosa 分析結果
            clap_result: CLAP get_formatted_tags() 返回的結構化結果
            normalized_features: 正規化後的特徵 (0-1 範圍)
            use_llm: 是否使用 LLM 生成自然語言描述
            llm_generator: LLM Generator 實例

        Returns:
            dict: 包含 prompt, tags, key, bpm, features, analysis_source
        """
        # 使用正規化特徵進行分類
        if normalized_features:
            norm = normalized_features
        else:
            # 如果沒有正規化特徵，使用原始特徵計算
            norm = self._normalize_features(features)

        bpm = features['bpm']
        key = features['key']
        mode = features.get('mode', 'major')

        # 1. 從正規化特徵生成 Librosa 標籤
        librosa_tags = self._generate_librosa_tags(norm, bpm)

        # 2. 處理 CLAP 標籤
        if clap_result and isinstance(clap_result, dict):
            clap_by_category = clap_result.get("by_category", {})
            clap_all_tags = clap_result.get("all_tags", [])
        else:
            # 兼容舊格式（純列表）
            clap_by_category = {}
            clap_all_tags = clap_result if isinstance(clap_result, list) else []

        # 3. 組合標籤（按類別）
        combined = self._combine_tags(librosa_tags, clap_by_category, clap_all_tags)

        # 4. 語義衝突檢測和去重
        cleaned = self._resolve_conflicts(combined)

        # 5. 生成 Prompt
        llm_prompt = None
        fallback_prompt = self._format_prompt(cleaned, key, mode, bpm)

        # 嘗試使用 LLM 生成自然語言描述
        if use_llm and llm_generator:
            try:
                llm_prompt = llm_generator.generate_prompt(
                    features=features,
                    normalized_features=norm,
                    tags_by_category=cleaned,
                    clap_tags=clap_all_tags
                )
                if llm_prompt:
                    logger.info(f"LLM generated prompt: {llm_prompt[:100]}...")
            except Exception as e:
                logger.warning(f"LLM generation failed, using fallback: {e}")

        # 選擇最終 prompt
        prompt_string = llm_prompt if llm_prompt else fallback_prompt

        # 收集所有標籤
        all_tags = []
        for category in ["genre", "mood", "instruments", "production"]:
            all_tags.extend(cleaned.get(category, []))

        return {
            "prompt": prompt_string,
            "prompt_fallback": fallback_prompt,  # 備援 prompt
            "prompt_source": "llm" if llm_prompt else "tags",
            "tags": all_tags,
            "tags_by_category": cleaned,
            "key": f"{key} {mode}",
            "bpm": bpm,
            "features": features,
            "normalized_features": norm,
            "analysis_source": "clap+librosa" if clap_all_tags else "librosa"
        }

    def _normalize_features(self, features: dict) -> dict:
        """將原始特徵正規化到 0-1（備用方法）"""
        brightness = features.get("brightness", 2000)
        norm_brightness = max(0, min((brightness - 1000) / 3000, 1.0))

        return {
            "bpm": features.get("bpm", 120),
            "key": features.get("key", "C"),
            "mode": features.get("mode", "major"),
            "energy": min(features.get("energy", 0) / 0.15, 1.0),
            "brightness": norm_brightness,
            "rhythm": min(features.get("rhythm", 0) / 1.5, 1.0),
            "contrast": min(features.get("contrast", 0) / 30, 1.0),
            "percussiveness": min(features.get("percussiveness", 0) / 0.15, 1.0),
        }

    def _generate_librosa_tags(self, norm: dict, bpm: int) -> dict:
        """
        從正規化特徵生成標籤（按類別組織）

        使用多特徵組合判斷，取代單一閾值
        """
        tags = {
            "genre": [],
            "mood": [],
            "instruments": [],
            "production": []
        }

        # 1. 強度分類（組合 energy, rhythm, contrast）
        intensity_level, intensity_tags = self._classify_intensity(norm)
        tags["mood"].extend(intensity_tags)

        # 2. 音色分類（組合 brightness, percussiveness）
        timbre_tags = self._classify_timbre(norm)
        # 分配到適當類別
        for tag in timbre_tags:
            if tag in ["electronic", "acoustic", "lo-fi"]:
                tags["production"].append(tag)
            elif tag in ["bright", "dark", "warm"]:
                tags["instruments"].append(tag)
            else:
                tags["production"].append(tag)

        # 3. 速度分類
        tempo_tags = self._classify_tempo(bpm, intensity_level)
        tags["mood"].extend(tempo_tags)

        # 4. 調性情緒
        mode_tags = self._classify_mode(norm.get("mode", "major"))
        tags["mood"].extend(mode_tags)

        # 5. 對比度/製作風格
        contrast_tags = self._classify_contrast(norm)
        tags["production"].extend(contrast_tags)

        return tags

    def _classify_intensity(self, norm: dict) -> tuple:
        """
        組合多特徵計算強度等級

        intensity = energy * 0.4 + rhythm * 0.35 + contrast * 0.25
        """
        energy = norm.get("energy", 0.5)
        rhythm = norm.get("rhythm", 0.5)
        contrast = norm.get("contrast", 0.5)

        score = energy * 0.4 + rhythm * 0.35 + contrast * 0.25

        if score > 0.7:
            return "high", ["powerful", "dynamic"]
        elif score > 0.4:
            return "medium", ["balanced"]
        else:
            return "low", ["soft", "gentle"]

    def _classify_timbre(self, norm: dict) -> list:
        """
        組合亮度和打擊感判斷音色
        """
        brightness = norm.get("brightness", 0.5)
        percussive = norm.get("percussiveness", 0.5)

        tags = []

        # 亮度 + 打擊感組合判斷
        if brightness > 0.7 and percussive < 0.3:
            tags.extend(["electronic", "bright", "modern"])
        elif brightness < 0.3 and percussive > 0.5:
            tags.extend(["lo-fi", "warm"])
        elif brightness < 0.3:
            tags.extend(["dark", "deep"])
        elif brightness > 0.6:
            tags.extend(["crisp", "polished"])
        else:
            tags.append("balanced tone")

        return tags

    def _classify_tempo(self, bpm: int, intensity_level: str) -> list:
        """根據 BPM 和強度分類速度感"""
        tags = []

        if bpm < 70:
            tags.append("ballad")
        elif bpm < 90:
            tags.append("downtempo")
        elif bpm < 110:
            if intensity_level == "low":
                tags.append("laid back")
            else:
                tags.append("groovy")
        elif bpm < 130:
            if intensity_level == "high":
                tags.append("driving")
            else:
                tags.append("upbeat")
        else:
            tags.append("high energy")

        return tags

    def _classify_mode(self, mode: str) -> list:
        """根據調性生成情緒標籤"""
        if mode == "minor":
            return ["emotional"]
        else:
            return ["uplifting"]

    def _classify_contrast(self, norm: dict) -> list:
        """根據對比度判斷製作風格"""
        contrast = norm.get("contrast", 0.5)

        if contrast > 0.7:
            return ["dynamic range"]
        elif contrast < 0.3:
            return ["compressed"]
        return []

    def _combine_tags(
        self,
        librosa_tags: dict,
        clap_by_category: dict,
        clap_all_tags: list
    ) -> dict:
        """
        組合 CLAP 和 Librosa 標籤

        CLAP 標籤優先（AI 語義分析），Librosa 標籤補充
        """
        combined = {
            "genre": [],
            "mood": [],
            "instruments": [],
            "production": []
        }

        # 類別映射（CLAP 類別 -> 我們的類別）
        category_map = {
            "genre": "genre",
            "mood": "mood",
            "instruments": "instruments",
            "vocals": "instruments",  # 合併到 instruments
            "production": "production",
            "rhythm": "mood",
            "era": "production",
            "atmosphere": "mood"
        }

        # 先加入 CLAP 標籤
        for clap_cat, our_cat in category_map.items():
            clap_tags = clap_by_category.get(clap_cat, [])
            for tag in clap_tags:
                if tag not in combined[our_cat]:
                    combined[our_cat].append(tag)

        # 補充 Librosa 標籤
        for category, tags in librosa_tags.items():
            for tag in tags:
                # 避免重複
                tag_lower = tag.lower()
                existing_lower = [t.lower() for t in combined[category]]
                if tag_lower not in existing_lower:
                    combined[category].append(tag)

        return combined

    def _resolve_conflicts(self, tags: dict) -> dict:
        """
        解決語義衝突和去重

        1. 檢測同一維度的衝突（如 high energy + low energy）
        2. 同組取最高優先級（第一個）
        3. 去除語義重複
        """
        resolved = {}

        for category, tag_list in tags.items():
            if not tag_list:
                resolved[category] = []
                continue

            # 1. 語義去重
            deduped = self._dedupe_synonyms(tag_list)

            # 2. 衝突檢測
            cleaned = self._detect_and_resolve_conflicts(deduped)

            # 限制每個類別的標籤數量
            max_per_category = {"genre": 2, "mood": 2, "instruments": 2, "production": 2}
            limit = max_per_category.get(category, 2)
            resolved[category] = cleaned[:limit]

        return resolved

    def _dedupe_synonyms(self, tag_list: list) -> list:
        """去除語義相似的標籤，保留第一個"""
        result = []
        used_groups = set()

        for tag in tag_list:
            tag_lower = tag.lower()

            # 檢查是否屬於已使用的同義詞組
            is_duplicate = False
            for canonical, synonyms in SEMANTIC_SYNONYMS.items():
                all_in_group = [canonical] + synonyms
                all_in_group_lower = [s.lower() for s in all_in_group]

                if tag_lower in all_in_group_lower:
                    if canonical in used_groups:
                        is_duplicate = True
                        break
                    else:
                        used_groups.add(canonical)
                        break

            if not is_duplicate and tag not in result:
                result.append(tag)

        return result

    def _detect_and_resolve_conflicts(self, tag_list: list) -> list:
        """
        檢測語義衝突，同一維度只保留第一個出現的

        例如: ["energetic", "ambient"] -> ["energetic"] (不保留 ambient)
        """
        result = []
        locked_dimensions = {}  # dimension -> "high" or "low"

        for tag in tag_list:
            tag_lower = tag.lower()
            has_conflict = False

            # 檢查所有衝突維度
            for dimension, levels in SEMANTIC_CONFLICTS.items():
                for level, level_tags in levels.items():
                    level_tags_lower = [t.lower() for t in level_tags]

                    if tag_lower in level_tags_lower:
                        # 該標籤屬於某個維度的某個等級
                        if dimension in locked_dimensions:
                            # 該維度已被鎖定
                            if locked_dimensions[dimension] != level:
                                # 衝突! 這個標籤與已選標籤衝突
                                has_conflict = True
                                break
                        else:
                            # 鎖定該維度
                            locked_dimensions[dimension] = level
                        break

                if has_conflict:
                    break

            if not has_conflict and tag not in result:
                result.append(tag)

        return result

    def _format_prompt(self, tags: dict, key: str, mode: str, bpm: int) -> str:
        """
        生成結構化 Prompt

        格式: [GENRE], [MOOD], [INSTRUMENTS], [PRODUCTION]; [KEY] [MODE], [BPM] bpm
        """
        parts = []

        # Genre
        genre = tags.get("genre", [])
        if genre:
            parts.append(" ".join(genre))

        # Mood
        mood = tags.get("mood", [])
        if mood:
            if len(mood) > 1:
                parts.append(f"{mood[0]} and {mood[1]}")
            else:
                parts.append(mood[0])

        # Instruments
        instruments = tags.get("instruments", [])
        if instruments:
            if len(instruments) > 1:
                parts.append(f"{instruments[0]} with {instruments[1]}")
            else:
                parts.append(instruments[0])

        # Production
        production = tags.get("production", [])
        if production:
            parts.append(f"{' '.join(production)} production")

        # 組合
        if parts:
            main_part = ", ".join(parts)
        else:
            main_part = "music"

        # 加入調性和 BPM
        return f"{main_part}; {key} {mode}, {bpm} bpm"
