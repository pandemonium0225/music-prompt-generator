# Music Prompt Generator - 未來優化路線圖

基於最新音樂 AI 研究（2024-2026），規劃以下優化方向。

---

## 🎯 短期目標 (v1.x)

### 1.1 改進現有分析精度

- [ ] **情緒效價 (Valence) 偵測**
  - 區分「快樂的快」vs「憤怒的快」
  - 使用預訓練的情緒分類模型
  - 參考: Spotify Audio Features API 的 valence 定義

- [ ] **喚醒度 (Arousal) 分析**
  - 測量音樂的「激昂程度」
  - 結合 RMS Energy + Spectral Flux + Tempo 綜合判斷

- [ ] **更精確的調性偵測**
  - 使用 Krumhansl-Schmuckler 演算法
  - 支援關係大小調判斷
  - 偵測轉調 (Key Change)

### 1.2 Prompt 品質提升

- [ ] **擴充風格標籤詞彙庫**
  - 建立 SUNO 官方支援的完整 tag 清單
  - 按曲風分類 (EDM, Jazz, Classical, Hip-hop...)
  - 加入時代風格 (80s synth, 90s grunge, 2010s trap...)

- [ ] **多語言 Prompt 支援**
  - 中文 → 英文風格詞彙對照表
  - 使用者可選擇輸出語言

---

## 🚀 中期目標 (v2.x) - CLAP 整合

### 2.1 CLAP 模型整合 (Contrastive Language-Audio Pretraining)

> CLAP 是目前最先進的「音訊-文字」跨模態模型，能直接從音訊生成高品質描述。

- [ ] **整合 LAION CLAP 或 Microsoft CLAP**
  - HuggingFace: `laion/clap-htsat-unfused`
  - 直接從音訊生成自然語言描述
  - 輸出範例: "A melancholic jazz piano with rain background"

- [ ] **混合分析模式**
  ```
  最終 Prompt = CLAP 描述 (語義) + Librosa 分析 (數據)
  ```
  - CLAP 提供高階風格描述
  - Librosa 提供精確數值 (BPM, Key)

- [ ] **技術實作**
  ```python
  # 預計新增 clap_analyzer.py
  from transformers import ClapModel, ClapProcessor

  class CLAPAnalyzer:
      def analyze(self, audio_path: str) -> str:
          # 返回自然語言描述
          pass
  ```

### 2.2 深度音樂資訊檢索 (Deep MIR)

- [ ] **樂器識別 (Instrument Recognition)**
  - 偵測主要樂器: 鋼琴、吉他、合成器、鼓...
  - 使用預訓練 CNN 模型

- [ ] **人聲偵測**
  - 判斷是否有人聲
  - 人聲類型: 男/女、獨唱/合唱

- [ ] **曲風分類 (Genre Classification)**
  - 多標籤分類: Pop, Rock, Electronic, Jazz...
  - 使用 GTZAN 或 FMA 資料集訓練的模型

---

## 🔬 長期目標 (v3.x) - 音訊條件生成

### 3.1 整合 MusicGen (Meta Audiocraft)

> MusicGen 支援 "Melody Conditioning"，可以保留原曲旋律並改變風格。

- [ ] **Melody Conditioning 功能**
  ```
  輸入: 使用者上傳的歌曲
  輸出: 旋律相似但風格不同的新音樂
  ```
  - 提取原曲的旋律輪廓 (Melody Profile)
  - 提取和弦行進 (Chord Progression)
  - 使用 MusicGen 以此為條件生成新曲

- [ ] **技術架構**
  ```
  User Audio → Melody Extraction → MusicGen → New Audio
                    ↓
              Style Prompt (from our analyzer)
  ```

- [ ] **API 整合選項**
  - Replicate API (雲端)
  - 本地部署 (需 GPU)

### 3.2 音軌分離分析 (Stem Separation)

> 使用 Demucs 將歌曲拆解，分別分析各軌道。

- [ ] **整合 Meta Demucs**
  - 分離: 人聲 / 鼓 / 貝斯 / 其他
  - 對每個音軌獨立分析

- [ ] **音軌級 Prompt 生成**
  ```json
  {
    "vocals": "female, airy, reverb",
    "drums": "808, trap, punchy",
    "bass": "sub-bass, sustained",
    "other": "synth pad, atmospheric"
  }
  ```

- [ ] **應用場景**
  - 更精確的風格重現
  - 部分替換 (例如: 保留人聲，換掉配樂)

---

## 🔮 探索性功能

### 4.1 Retrieval-Augmented Generation (RAG) for Music

- [ ] **建立參考音樂資料庫**
  - 儲存分析過的歌曲特徵
  - 支援「找類似的歌」功能

- [ ] **Prompt 優化建議**
  - 基於相似歌曲的成功 Prompt
  - A/B 測試哪些 tag 組合效果最好

### 4.2 使用者偏好學習

- [ ] **品味模型 (Taste Profile)**
  - 分析使用者上傳的多首歌曲
  - 建立個人化的風格向量

- [ ] **推薦式 Prompt 生成**
  - "根據您的品味，建議使用這個 Prompt..."

### 4.3 即時音訊串流分析

- [ ] **支援麥克風輸入**
  - 即時哼唱 → 即時生成 Prompt

- [ ] **支援 YouTube/Spotify 連結**
  - 直接分析線上音樂 (需處理版權問題)

---

## 📚 參考資源

### 學術論文
- "Enhancing Text-to-Music Generation through Retrieval-Augmented Prompt Rewrite" (2025)
- "CLAP: Learning Audio Concepts from Natural Language Supervision" (2023)
- "MusicGen: Simple and Controllable Music Generation" (Meta, 2023)

### 開源模型
- [LAION CLAP](https://huggingface.co/laion/clap-htsat-unfused)
- [Meta MusicGen](https://huggingface.co/facebook/musicgen-melody)
- [Meta Demucs](https://github.com/facebookresearch/demucs)

### 資料集
- [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html)
- [Free Music Archive (FMA)](https://github.com/mdeff/fma)
- [MusicCaps](https://huggingface.co/datasets/google/MusicCaps)

---

## 📋 優先級排序

| 優先級 | 功能 | 難度 | 影響力 |
|--------|------|------|--------|
| P0 | 情緒效價偵測 | ⭐⭐ | ⭐⭐⭐ |
| P0 | 擴充風格標籤詞彙庫 | ⭐ | ⭐⭐⭐ |
| P1 | CLAP 整合 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| P1 | 曲風分類 | ⭐⭐ | ⭐⭐⭐⭐ |
| P2 | MusicGen Melody Conditioning | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| P2 | Demucs 音軌分離 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| P3 | RAG 音樂資料庫 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| P3 | 使用者偏好學習 | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

*最後更新: 2026-01-18*
