"""Audio Sensory Encoder.

Phase 4+ sensory bandwidth improvement B: audio modality input.

Provides a lightweight mel-spectrogram encoder for processing audio
input. Designed to be compact (~0.5 GB VRAM max) and optional —
trains without audio if not available.

Architecture:
    - Raw audio → Mel spectrogram (via torchaudio or numpy fallback)
    - Small CNN: (C, Freq, Time) → conv blocks → global pool → embedding
    - Output: (d_model,) vector for fusion with visual/touch modalities

Bounded: fixed CNN size, fixed spectrogram dimensions.

音频编码器：梅尔频谱 → 小型 CNN → 嵌入向量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


_AUDIO_AVAILABLE = False
try:
    import torchaudio
    _AUDIO_AVAILABLE = True
except ImportError:
    pass


@dataclass
class AudioEncoderConfig:
    """Configuration for :class:`AudioEncoder`.

    - ``sample_rate``: expected audio sample rate (Hz).
    - ``n_mels``: number of mel filterbanks.
    - ``n_fft``: FFT window size.
    - ``hop_length``: hop length between frames.
    - ``max_duration_s``: maximum audio duration in seconds.
    - ``d_model``: output embedding dimension.
    - ``hidden``: CNN hidden channel count.
    """

    sample_rate: int = 16000
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160
    max_duration_s: float = 3.0
    d_model: int = 128
    hidden: int = 64


class AudioEncoder(nn.Module):
    """Lightweight audio → embedding encoder.

    Converts raw audio waveforms to mel spectrograms, then processes
    through a small CNN to produce a fixed-size embedding vector.

    Bounded: spectrum dimensions fixed by config, CNN size fixed.
    VRAM: ~0.1 GB (64 mel bands × 300 frames × 64 channels).

    Usage:
        encoder = AudioEncoder(config)
        emb = encoder(waveform)  # waveform: (B, num_samples)
    """

    def __init__(
        self,
        config: AudioEncoderConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AudioEncoderConfig()
        cfg = self.config
        self._available = _AUDIO_AVAILABLE

        # Max time frames from max_duration_s
        max_samples = int(cfg.sample_rate * cfg.max_duration_s)
        n_frames = (max_samples - cfg.n_fft) // cfg.hop_length + 1
        if n_frames <= 0:
            n_frames = (
                max(1, int(cfg.sample_rate * 1.0) - cfg.n_fft) // cfg.hop_length + 1
            )

        # Small CNN: mel bands → hidden → d_model
        self._cnn = nn.Sequential(
            # Block 1: (B, 1, n_mels, n_frames) → (B, hidden, n_mels/2, n_frames/2)
            nn.Conv2d(1, cfg.hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cfg.hidden),
            nn.GELU(),
            # Block 2: halve again
            nn.Conv2d(cfg.hidden, cfg.hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cfg.hidden * 2),
            nn.GELU(),
            # Block 3: final conv
            nn.Conv2d(cfg.hidden * 2, cfg.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cfg.d_model),
            nn.GELU(),
        )

        # Global average pool → d_model
        self._output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        if _AUDIO_AVAILABLE:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
            )
        else:
            self._mel_transform = None

    @property
    def capacity(self) -> int:
        return 1  # fixed-size

    def __len__(self) -> int:
        return 0

    @property
    def is_available(self) -> bool:
        return self._available

    # -------------------------------------------------------- encode

    def _to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to mel spectrogram.

        Args:
            waveform: (B, num_samples) or (num_samples,)

        Returns:
            (B, 1, n_mels, n_frames) mel spectrogram.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if self._mel_transform is not None:
            self._mel_transform.to(waveform.device)
            mel = self._mel_transform(waveform)  # (B, n_mels, n_frames)
        else:
            # NumPy fallback: simple STFT approximation
            mel = self._numpy_mel_approx(waveform)

        # Log scale + normalize
        mel = torch.log(mel.clamp(min=1e-5))
        mel = (mel - mel.mean(dim=(-2, -1), keepdim=True)) / (
            mel.std(dim=(-2, -1), keepdim=True) + 1e-8
        )
        return mel.unsqueeze(1)  # (B, 1, n_mels, n_frames)

    def _numpy_mel_approx(self, waveform: torch.Tensor) -> torch.Tensor:
        """NumPy fallback mel-like spectrogram (no torchaudio)."""
        import numpy as np
        cfg = self.config
        wav = waveform.cpu().numpy()
        B = wav.shape[0]

        n_frames_max = (wav.shape[1] - cfg.n_fft) // cfg.hop_length + 1
        if n_frames_max <= 0:
            n_frames_max = 1

        mel_specs = np.zeros((B, cfg.n_mels, n_frames_max), dtype=np.float32)
        window = np.hanning(cfg.n_fft)

        for b in range(B):
            for i in range(n_frames_max):
                start = i * cfg.hop_length
                end = start + cfg.n_fft
                if end > wav.shape[1]:
                    break
                frame = wav[b, start:end] * window
                spec = np.abs(np.fft.rfft(frame, n=cfg.n_fft))
                # Simple mel-like: average into n_mels bands (log-spaced)
                spec_half = spec[: len(spec) // 2 + 1]
                band_size = max(1, len(spec_half) // cfg.n_mels)
                for j in range(cfg.n_mels):
                    s = j * band_size
                    e = min(s + band_size, len(spec_half))
                    mel_specs[b, j, i] = spec_half[s:e].mean()

        return torch.from_numpy(mel_specs).to(waveform.device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to embedding.

        Args:
            waveform: (B, num_samples) raw audio.

        Returns:
            (B, d_model) audio embedding.
        """
        mel = self._to_mel(waveform)
        feat = self._cnn(mel)
        emb = self._output(feat)
        return emb

    # -------------------------------------------------------- persistence

    def summary(self) -> dict:
        params = sum(p.numel() for p in self.parameters())
        return {
            "available": self._available,
            "params": params,
            "d_model": self.config.d_model,
            "n_mels": self.config.n_mels,
        }
