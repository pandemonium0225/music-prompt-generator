"""
LLM Prompt Generator

使用 OpenAI GPT 將音訊分析結果轉換為自然語言的 SUNO 風格描述
"""

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# 環境變數配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"


# System prompt for GPT
SYSTEM_PROMPT = """You are a music description expert specializing in creating prompts for SUNO AI music generation.

Your task is to write a natural, flowing description of music based on audio analysis data. The description should:

1. Be written in English as 1-2 concise sentences
2. Describe the music's style, mood, instruments, and feel naturally
3. Be suitable as a SUNO AI prompt (focus on style/mood/instruments, not technical details)
4. Sound like a human describing the music, not a list of tags
5. Include the key and BPM naturally at the end

Good examples:
- "A dreamy lo-fi hip hop track with mellow piano chords and soft vinyl crackle, creating a nostalgic late-night study vibe. Key of C minor, 85 BPM."
- "Energetic electronic dance music with pulsing synths and driving four-on-the-floor beats, building to an euphoric drop. A major, 128 BPM."
- "Intimate acoustic ballad featuring gentle fingerpicked guitar and heartfelt vocals, evoking bittersweet emotions. G major, 72 BPM."

Bad examples (avoid these):
- "pop, electronic, upbeat, synth, drums, 120 bpm" (just tags, not natural language)
- "This song has energy level 0.5 and brightness 2000 Hz" (too technical)
- "A song that is happy and sad at the same time" (contradictory)"""


USER_PROMPT_TEMPLATE = """Based on the following audio analysis, write a natural music description for SUNO AI:

## Audio Features
- BPM: {bpm}
- Key: {key} {mode}
- Energy Level: {energy_desc} ({energy:.0%})
- Brightness: {brightness_desc} ({brightness:.0%})
- Rhythm Intensity: {rhythm_desc} ({rhythm:.0%})
- Dynamic Range: {dynamics_desc} ({contrast:.0%})

## Detected Tags
{tags_section}

## Instructions
Write a 1-2 sentence natural description that captures this music's essence. End with the key and BPM.
Do NOT just list tags - write flowing, descriptive sentences that a human would use to describe the music."""


class LLMGenerator:
    """
    使用 OpenAI GPT 生成自然語言音樂描述
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化 LLM Generator

        Args:
            api_key: OpenAI API key (預設從環境變數讀取)
            model: 使用的模型 (預設 gpt-4o-mini)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        self.client = None

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"LLM Generator initialized with model: {self.model}")
        else:
            logger.warning("OpenAI API key not set, LLM generation disabled")

    def generate_prompt(
        self,
        features: dict,
        normalized_features: dict,
        tags_by_category: Optional[dict] = None,
        clap_tags: Optional[list] = None
    ) -> Optional[str]:
        """
        使用 GPT 生成自然語言 prompt

        Args:
            features: 原始音訊特徵
            normalized_features: 正規化特徵 (0-1)
            tags_by_category: 按類別分組的標籤
            clap_tags: CLAP 分析的標籤列表

        Returns:
            str: 自然語言的音樂描述，或 None 如果生成失敗
        """
        if not self.client:
            logger.warning("LLM client not initialized")
            return None

        try:
            # 準備特徵描述
            energy = normalized_features.get("energy", 0.5)
            brightness = normalized_features.get("brightness", 0.5)
            rhythm = normalized_features.get("rhythm", 0.5)
            contrast = normalized_features.get("contrast", 0.5)

            # 將數值轉換為描述性文字
            energy_desc = self._level_to_desc(energy, ["soft and gentle", "moderate", "energetic and powerful"])
            brightness_desc = self._level_to_desc(brightness, ["warm and dark", "balanced", "bright and crisp"])
            rhythm_desc = self._level_to_desc(rhythm, ["relaxed", "steady", "driving and intense"])
            dynamics_desc = self._level_to_desc(contrast, ["compressed", "moderate dynamics", "wide dynamic range"])

            # 整理標籤
            tags_section = self._format_tags(tags_by_category, clap_tags)

            # 組合 user prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(
                bpm=features.get("bpm", 120),
                key=features.get("key", "C"),
                mode=features.get("mode", "major"),
                energy=energy,
                energy_desc=energy_desc,
                brightness=brightness,
                brightness_desc=brightness_desc,
                rhythm=rhythm,
                rhythm_desc=rhythm_desc,
                contrast=contrast,
                dynamics_desc=dynamics_desc,
                tags_section=tags_section
            )

            logger.debug(f"Sending prompt to GPT: {user_prompt[:200]}...")

            # 呼叫 OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )

            result = response.choices[0].message.content.strip()
            logger.info(f"GPT generated prompt: {result}")

            return result

        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return None

    def _level_to_desc(self, value: float, descriptions: list) -> str:
        """將 0-1 的數值轉換為描述文字"""
        if value < 0.35:
            return descriptions[0]
        elif value < 0.65:
            return descriptions[1]
        else:
            return descriptions[2]

    def _format_tags(self, tags_by_category: Optional[dict], clap_tags: Optional[list]) -> str:
        """格式化標籤為文字"""
        lines = []

        if tags_by_category:
            for category, tags in tags_by_category.items():
                if tags:
                    lines.append(f"- {category.title()}: {', '.join(tags)}")

        if clap_tags and not tags_by_category:
            lines.append(f"- Detected: {', '.join(clap_tags[:8])}")

        if not lines:
            return "- No specific tags detected"

        return "\n".join(lines)


# 單例模式
_llm_instance: Optional[LLMGenerator] = None


def get_llm_generator() -> Optional[LLMGenerator]:
    """取得 LLM Generator 單例"""
    global _llm_instance

    if not LLM_ENABLED:
        return None

    if _llm_instance is None:
        _llm_instance = LLMGenerator()

    return _llm_instance
