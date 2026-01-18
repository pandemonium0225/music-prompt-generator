class SunoTranslator:
    """
    將音訊分析結果轉換為 SUNO 風格的 Prompt
    """

    def generate_prompt(self, features: dict) -> dict:
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

        # 去除重複並組合
        all_tags = list(dict.fromkeys(style_tags + mood_tags + instrument_tags + tags))

        # 限制 tag 數量，保持 prompt 精簡
        selected_tags = all_tags[:8]

        # 組合最終 Prompt
        prompt_string = f"{', '.join(selected_tags)}, {key} {mode}, {bpm} bpm"

        return {
            "prompt": prompt_string,
            "tags": selected_tags,
            "key": f"{key} {mode}",
            "bpm": bpm,
            "features": features
        }
