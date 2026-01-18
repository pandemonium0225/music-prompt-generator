import logging
import warnings
from typing import Union

import librosa
import numpy as np

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """音訊分析器 - 使用 Librosa 提取音樂特徵"""

    # 標準音高名稱
    KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, sample_rate: int = 22050, duration: int = 60):
        """
        初始化音訊分析器

        Args:
            sample_rate: 取樣率 (預設 22050 Hz)
            duration: 分析時長上限 (預設 60 秒)
        """
        self.sr = sample_rate
        self.duration = duration

    def analyze(self, file_path: str) -> Union[dict, None]:
        """
        分析音訊檔案，提取音樂特徵

        Args:
            file_path: 音訊檔案路徑

        Returns:
            dict: 包含 bpm, energy, brightness, rhythm, key, mode 等特徵
            None: 分析失敗時返回
        """
        try:
            logger.debug(f"Loading audio file: {file_path}")

            # 載入音訊 (限制時長以提升效能)
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)

            if len(y) == 0:
                logger.error("Empty audio file")
                return None

            logger.debug(f"Audio loaded: {len(y)} samples, {sr} Hz")

            # 1. BPM (節奏速度)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = int(round(float(tempo)))

            # 2. 能量動態 (RMS Energy) - 0.0 ~ 0.5+
            rms = librosa.feature.rms(y=y)
            avg_energy = float(np.mean(rms))
            energy_std = float(np.std(rms))

            # 3. 音色明亮度 (Spectral Centroid) - 1000 ~ 4000+ Hz
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            brightness = float(np.mean(cent))

            # 4. 節奏強度 (Onset Strength)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            rhythm_strength = float(np.mean(onset_env))

            # 5. 調性偵測 (Key + Mode)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            key_idx = int(np.argmax(chroma_mean))
            detected_key = self.KEYS[key_idx]

            # 估計大調/小調 (比較大三和弦與小三和弦的強度)
            major_strength = (
                chroma_mean[key_idx] +
                chroma_mean[(key_idx + 4) % 12] +
                chroma_mean[(key_idx + 7) % 12]
            )
            minor_strength = (
                chroma_mean[key_idx] +
                chroma_mean[(key_idx + 3) % 12] +
                chroma_mean[(key_idx + 7) % 12]
            )
            mode = "major" if major_strength >= minor_strength else "minor"

            # 6. 頻譜對比度 (音樂層次感)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            avg_contrast = float(np.mean(contrast))

            # 7. 零交叉率 (打擊樂/噪音成分指標)
            zcr = librosa.feature.zero_crossing_rate(y)
            avg_zcr = float(np.mean(zcr))

            result = {
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

            logger.debug(f"Analysis result: {result}")
            return result

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return None
