import React, { useState, useCallback } from 'react';
import axios from 'axios';
import {
  Upload,
  Copy,
  Check,
  Music2,
  ArrowRight,
  Loader2,
  FileAudio,
  Sparkles,
  AlertCircle
} from 'lucide-react';

// API URL - 開發環境使用 proxy，生產環境使用環境變數
const API_URL = import.meta.env.VITE_API_URL || '/api';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  // 處理拖放
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith('audio/') ||
          droppedFile.name.match(/\.(mp3|wav|flac|ogg)$/i)) {
        setFile(droppedFile);
        setError(null);
        setResult(null);
      } else {
        setError('請上傳有效的音訊檔案 (MP3, WAV, FLAC, OGG)');
      }
    }
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_URL}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 秒超時
      });

      if (res.data.success) {
        setResult(res.data.data);
      } else {
        setError(res.data.detail || '分析失敗');
      }
    } catch (err) {
      console.error('Upload error:', err);
      if (err.code === 'ECONNABORTED') {
        setError('分析超時，請嘗試較小的檔案');
      } else if (err.response) {
        setError(err.response.data.detail || '分析失敗');
      } else if (err.request) {
        setError('無法連接到伺服器，請稍後再試');
      } else {
        setError('發生未知錯誤');
      }
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async () => {
    if (!result) return;

    const text = result.prompt;

    try {
      // 嘗試使用現代 Clipboard API
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        // Fallback: 使用傳統方法 (支援非安全環境如 WSL IP)
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        textArea.style.top = '-9999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Copy failed:', err);
      // 顯示手動複製提示
      alert('自動複製失敗，請手動選取文字後按 Ctrl+C 複製');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 font-sans p-4 sm:p-6 flex flex-col items-center justify-center">

      {/* Header */}
      <header className="mb-8 sm:mb-10 text-center max-w-2xl">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Sparkles className="w-8 h-8 text-amber-500" />
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold bg-gradient-to-r from-amber-200 via-yellow-400 to-amber-600 bg-clip-text text-transparent">
            SUNO Prompt Architect
          </h1>
        </div>
        <p className="text-neutral-400 text-base sm:text-lg">
          上傳歌曲 → 解析音樂 DNA → 取得 SUNO 提示詞
        </p>
      </header>

      <main className="w-full max-w-3xl space-y-6">

        {/* 上傳區 */}
        <div
          className={`relative bg-neutral-900/50 border-2 border-dashed rounded-2xl p-8 sm:p-10 text-center transition-all duration-300 cursor-pointer group overflow-hidden
            ${dragActive
              ? 'border-amber-500 bg-amber-500/10'
              : file
                ? 'border-amber-500/50 hover:border-amber-500'
                : 'border-neutral-700 hover:border-neutral-500'
            }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="audio/*,.mp3,.wav,.flac,.ogg"
            onChange={handleFileChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
          />

          <div className="relative z-10 flex flex-col items-center gap-4">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300
              ${file
                ? 'bg-amber-500/20 scale-110'
                : 'bg-neutral-800 group-hover:bg-neutral-700 group-hover:scale-105'
              }`}
            >
              {file ? (
                <FileAudio className="w-8 h-8 text-amber-500" />
              ) : (
                <Upload className="w-8 h-8 text-neutral-500 group-hover:text-neutral-400" />
              )}
            </div>

            <div>
              <h3 className="text-lg sm:text-xl font-bold text-white mb-1">
                {file ? file.name : '拖放或點擊上傳音訊檔案'}
              </h3>
              <p className="text-neutral-500 text-sm">
                {file
                  ? `檔案大小: ${formatFileSize(file.size)}`
                  : '支援 MP3, WAV, FLAC, OGG (建議 < 10MB)'
                }
              </p>
            </div>
          </div>
        </div>

        {/* 錯誤訊息 */}
        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* 分析按鈕 */}
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className={`w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-3 transition-all duration-300
            ${loading
              ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
              : !file
                ? 'bg-neutral-800 text-neutral-600 cursor-not-allowed'
                : 'bg-amber-500 hover:bg-amber-400 text-black shadow-lg shadow-amber-500/20 hover:shadow-amber-500/40 active:scale-[0.98]'
            }`}
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              正在解析波形結構...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              生成 Prompt
            </>
          )}
        </button>

        {/* 結果區 */}
        {result && (
          <div className="animate-fade-in-up space-y-6">

            {/* Prompt 卡片 */}
            <div className="bg-neutral-900 rounded-xl overflow-hidden border border-amber-500/30 shadow-2xl shadow-amber-500/5">
              <div className="bg-gradient-to-r from-amber-500/10 to-transparent p-4 border-b border-neutral-800">
                <div className="flex justify-between items-center">
                  <span className="text-amber-500 text-xs font-bold tracking-widest uppercase flex items-center gap-2">
                    <Music2 className="w-4 h-4" />
                    SUNO Style Prompt
                  </span>
                  <button
                    onClick={copyToClipboard}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-neutral-400 hover:text-white hover:bg-neutral-800 transition-all"
                  >
                    {copied ? (
                      <>
                        <Check className="w-4 h-4 text-green-500" />
                        <span className="text-green-500">已複製</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4" />
                        複製
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="p-6">
                <p className="text-xl sm:text-2xl font-mono text-white leading-relaxed break-words select-all bg-neutral-950/50 p-4 rounded-lg border border-neutral-800">
                  {result.prompt}
                </p>

                {/* 特徵數據 */}
                <div className="mt-6 pt-6 border-t border-neutral-800">
                  <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-4">音樂特徵分析</h4>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <FeatureCard label="調性" value={result.key} />
                    <FeatureCard label="BPM" value={result.bpm} />
                    <FeatureCard
                      label="能量"
                      value={`${(result.features.energy * 100).toFixed(0)}%`}
                    />
                    <FeatureCard
                      label="明亮度"
                      value={result.features.brightness > 2500 ? '明亮' : result.features.brightness < 1500 ? '溫暖' : '中性'}
                    />
                  </div>
                </div>

                {/* Tags */}
                <div className="mt-4 pt-4 border-t border-neutral-800">
                  <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-3">風格標籤</h4>
                  <div className="flex flex-wrap gap-2">
                    {result.tags.map((tag, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 bg-neutral-800 text-neutral-300 rounded-full text-sm"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* SUNO 連結 */}
            <a
              href="https://suno.com/create"
              target="_blank"
              rel="noreferrer"
              className="flex items-center justify-between p-5 bg-neutral-900 hover:bg-neutral-800 rounded-xl border border-neutral-800 hover:border-neutral-600 transition-all group"
            >
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full bg-black flex items-center justify-center group-hover:scale-105 transition-transform">
                  <Music2 className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h4 className="font-bold text-white">前往 SUNO 創作</h4>
                  <p className="text-sm text-neutral-500">將 Prompt 貼上至 Style of Music</p>
                </div>
              </div>
              <ArrowRight className="text-neutral-600 group-hover:text-white group-hover:translate-x-1 transition-all" />
            </a>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 text-center text-neutral-600 text-sm">
        <p>Music Prompt Generator v1.0</p>
      </footer>
    </div>
  );
}

// 特徵卡片元件
function FeatureCard({ label, value }) {
  return (
    <div className="bg-neutral-950/50 rounded-lg p-3 text-center border border-neutral-800">
      <div className="text-xs text-neutral-500 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-lg font-bold text-neutral-200">{value}</div>
    </div>
  );
}

export default App;
