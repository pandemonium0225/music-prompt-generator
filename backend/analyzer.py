import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class AudioAnalyzer:
    def __init__(self):
        self.sr = 22050
        self.keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def analyze(self, file_path: str) -> dict | None:
        """
        分析音訊檔案，提取音樂特徵

        Returns:
            dict: 包含 bpm, energy, brightness, rhythm, key, mode 等特徵
        """
        try:
            # 為了效能，只讀取前 60 秒
            y, sr = librosa.load(file_path, sr=self.sr, duration=60)

            # 1. BPM (節奏速度)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = int(round(float(tempo)))

            # 2. 能量動態 (RMS Energy) - 0.0 ~ 0.5+
            rms = librosa.feature.rms(y=y)
            avg_energy = float(np.mean(rms))
            energy_std = float(np.std(rms))  # 能量變化程度

            # 3. 音色明亮度 (Spectral Centroid) - 1000 ~ 4000+
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            brightness = float(np.mean(cent))

            # 4. 節奏強度 (Onset Strength)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            rhythm_strength = float(np.mean(onset_env))

            # 5. 調性偵測 (Key + Mode)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            key_idx = int(np.argmax(chroma_mean))
            detected_key = self.keys[key_idx]

            # 估計大調/小調 (簡化版：比較大三和弦與小三和弦的強度)
            major_strength = chroma_mean[key_idx] + chroma_mean[(key_idx + 4) % 12] + chroma_mean[(key_idx + 7) % 12]
            minor_strength = chroma_mean[key_idx] + chroma_mean[(key_idx + 3) % 12] + chroma_mean[(key_idx + 7) % 12]
            mode = "major" if major_strength >= minor_strength else "minor"

            # 6. 頻譜對比度 (用於判斷音樂的層次感)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            avg_contrast = float(np.mean(contrast))

            # 7. 零交叉率 (用於判斷是否有較多打擊樂/噪音成分)
            zcr = librosa.feature.zero_crossing_rate(y)
            avg_zcr = float(np.mean(zcr))

            return {
                "bpm": bpm,
                "energy": round(avg_energy, 4),
                "energy_variation": round(energy_std, 4),
                "brightness": round(brightness, 2),
                "rhythm": round(rhythm_strength, 4),
                "key": detected_key,
                "mode": mode,
                "contrast": round(avg_contrast, 4),
                "percussiveness": round(avg_zcr, 4)
            }

        except Exception as e:
            print(f"Analysis Error: {e}")
            return None
