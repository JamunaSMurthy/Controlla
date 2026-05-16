"""Audio feature extraction wrappers for CREMA-D, IEMOCAP, and related corpora."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_audio_waveform(audio_path: str | Path) -> tuple[int, np.ndarray]:
    """Load a WAV file as a mono float32 waveform in the range [-1, 1]."""
    path = Path(audio_path)
    with wave.open(str(path), "rb") as handle:
        num_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        num_frames = handle.getnframes()
        raw_frames = handle.readframes(num_frames)

    if sample_width == 1:
        waveform = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32)
        waveform = (waveform - 128.0) / 128.0
    elif sample_width == 2:
        waveform = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        waveform = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if num_channels > 1:
        waveform = waveform.reshape(-1, num_channels).mean(axis=1)
    return sample_rate, waveform.astype(np.float32)


def _frame_waveform(waveform: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if waveform.size == 0:
        return np.zeros((1, frame_size), dtype=np.float32)
    if waveform.size < frame_size:
        padded = np.pad(waveform, (0, frame_size - waveform.size))
        return padded[None, :].astype(np.float32)

    num_frames = 1 + int(np.ceil((waveform.size - frame_size) / hop_size))
    total_length = frame_size + hop_size * (num_frames - 1)
    padded = np.pad(waveform, (0, max(0, total_length - waveform.size)))
    frames = [padded[index * hop_size : index * hop_size + frame_size] for index in range(num_frames)]
    return np.stack(frames).astype(np.float32)


def _resample_vector(values: np.ndarray, target_size: int) -> np.ndarray:
    if values.size == target_size:
        return values.astype(np.float32)
    if values.size == 1:
        return np.full(target_size, float(values[0]), dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_size, dtype=np.float32)
    resampled = np.interp(target_positions, source_positions, values.astype(np.float32))
    return resampled.astype(np.float32)


def extract_audio_embedding(audio_path: str | Path, embedding_dim: int = 64) -> tuple[np.ndarray, dict[str, float]]:
    sample_rate, waveform = load_audio_waveform(audio_path)
    if waveform.size == 0:
        waveform = np.zeros(1, dtype=np.float32)

    duration_sec = float(waveform.size / sample_rate) if sample_rate else 0.0
    frame_size = max(128, int(sample_rate * 0.025)) if sample_rate else 400
    hop_size = max(64, int(sample_rate * 0.010)) if sample_rate else 160
    frames = _frame_waveform(waveform, frame_size, hop_size)

    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-8)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1).astype(np.float32)

    fft_size = int(2 ** np.ceil(np.log2(frame_size)))
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_size), n=fft_size, axis=1)).astype(np.float32)
    mean_spectrum = np.mean(spectrum, axis=0)
    mean_spectrum = np.log1p(mean_spectrum)

    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate if sample_rate else 1.0).astype(np.float32)
    spectral_mass = np.sum(mean_spectrum) + 1e-8
    spectral_centroid = float(np.sum(freqs * mean_spectrum) / spectral_mass)
    spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mean_spectrum) / spectral_mass))

    summary_vector = np.array(
        [
            float(np.mean(waveform)),
            float(np.std(waveform)),
            float(np.max(np.abs(waveform))),
            float(np.mean(np.abs(waveform))),
            float(np.min(waveform)),
            float(np.max(waveform)),
            float(np.mean(rms)),
            float(np.std(rms)),
            float(np.min(rms)),
            float(np.max(rms)),
            float(np.mean(zcr)),
            float(np.std(zcr)),
            duration_sec,
            spectral_centroid,
            spectral_bandwidth,
            float(sample_rate),
        ],
        dtype=np.float32,
    )

    spectral_vector = _resample_vector(mean_spectrum, max(embedding_dim - summary_vector.size, 1))
    embedding = np.concatenate([summary_vector, spectral_vector], axis=0)
    embedding = _resample_vector(embedding, embedding_dim)

    metadata = {
        "sample_rate": float(sample_rate),
        "duration_sec": duration_sec,
        "num_samples": float(waveform.size),
        "embedding_dim": float(embedding_dim),
    }
    return embedding.astype(np.float32), metadata


def extract_audio_features(audio_path: str, output_path: str | Path | None = None) -> str:
    """Extract and persist a deterministic audio embedding."""
    embedding, _ = extract_audio_embedding(audio_path)
    destination = Path(output_path) if output_path is not None else Path(audio_path).with_suffix(".audio.npy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, embedding)
    return str(destination.resolve())