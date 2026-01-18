from typing import Optional


class SunoTranslator:
    """
    將音訊分析結果轉換為 SUNO 風格的 Prompt

    支援兩種分析來源:
    - Librosa: 數值特徵 (BPM, Energy, Key...)
    - CLAP: 語義標籤 (AI 生成的風格描述)
    """

    def generate_prompt(
        self, features: dict, clap_tags: Optional[list] = None
    ) -> dict:
        bpm = features['bpm']
        energy = features['energy']
        brightness = features['brightness']
        rhythm = features['rhythm']
        key = features['key']
        mode = features.get('mode', 'major')
        contrast = features.get('contrast', 0)
        percussiveness = features.get('percussiveness', 0)

        tags = []
        style_tags = []
        mood_tags = []
        instrument_tags = []

        # 1. 速度與律動 (Tempo)
        if bpm < 70:
            tags.extend(["very slow", "ballad"])
            mood_tags.append("contemplative")
        elif bpm < 90:
            tags.extend(["downtempo", "slow burn"])
            mood_tags.append("relaxed")
        elif bpm < 110:
            tags.extend(["mid-tempo", "groovy"])
        elif bpm < 130:
            tags.extend(["upbeat", "driving beat"])
            mood_tags.append("energetic")
        elif bpm < 150:
            tags.extend(["fast-paced", "high energy"])
            mood_tags.append("intense")
        else:
            tags.extend(["very fast", "frenetic"])
            mood_tags.append("explosive")

        # 2. 能量與強度 (Energy + Rhythm)
        if energy > 0.2 and rhythm > 1.5:
            mood_tags.extend(["powerful", "anthemic"])
            style_tags.append("stadium")
        elif energy > 0.15 and rhythm > 1.2:
            mood_tags.extend(["dynamic", "punchy"])
        elif energy < 0.03:
            mood_tags.extend(["minimalist", "ethereal", "ambient"])
            style_tags.append("atmospheric")
        elif energy < 0.06:
            mood_tags.extend(["soft", "gentle", "intimate"])
        else:
            mood_tags.append("balanced")

        # 3. 音色與樂器 (Brightness)
        if brightness > 3500:
            instrument_tags.extend(["electronic", "synthesizer", "bright"])
            style_tags.append("modern")
        elif brightness > 2500:
            instrument_tags.extend(["crisp", "polished"])
            style_tags.append("pop")
        elif brightness < 1200:
            instrument_tags.extend(["deep bass", "sub-heavy", "dark"])
            style_tags.append("underground")
        elif brightness < 1800:
            instrument_tags.extend(["warm", "lo-fi", "vintage"])
            style_tags.append("acoustic")
        else:
            instrument_tags.extend(["clean mix", "balanced tone"])

        # 4. 調性 (Key + Mode)
        if mode == "minor":
            mood_tags.extend(["melancholic", "emotional"])
        else:
            mood_tags.extend(["uplifting", "bright mood"])

        # 5. 打擊感 (Percussiveness)
        if percussiveness > 0.15:
            instrument_tags.append("percussive")
            style_tags.append("rhythmic")
        elif percussiveness < 0.05:
            instrument_tags.append("smooth")
            style_tags.append("flowing")

        # 6. 對比度 (Contrast - 音樂層次)
        if contrast > 25:
            style_tags.append("dynamic range")
            mood_tags.append("cinematic")
        elif contrast < 10:
            style_tags.append("compressed")
            mood_tags.append("consistent")

        # 去除重複並組合 Librosa 標籤
        librosa_tags = list(dict.fromkeys(style_tags + mood_tags + instrument_tags + tags))

        # 整合 CLAP 標籤 (如果有)
        if clap_tags:
            # CLAP 標籤優先，因為是 AI 語義分析結果
            # 避免重複: 先加入 CLAP 標籤，再補充 Librosa 標籤
            combined_tags = list(clap_tags)  # CLAP tags first
            clap_lower = {t.lower() for t in clap_tags}

            for tag in librosa_tags:
                if tag.lower() not in clap_lower:
                    combined_tags.append(tag)

            # 限制 tag 數量，保持 prompt 精簡 (CLAP 提供更多有意義的標籤)
            selected_tags = combined_tags[:10]
            analysis_source = "clap+librosa"
        else:
            # 僅使用 Librosa 分析
            selected_tags = librosa_tags[:8]
            analysis_source = "librosa"

        # 組合最終 Prompt
        prompt_string = f"{', '.join(selected_tags)}, {key} {mode}, {bpm} bpm"

        return {
            "prompt": prompt_string,
            "tags": selected_tags,
            "key": f"{key} {mode}",
            "bpm": bpm,
            "features": features,
            "analysis_source": analysis_source
        }
