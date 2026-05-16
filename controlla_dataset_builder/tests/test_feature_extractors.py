"""Step 5 tests for lightweight audio and identity feature extraction."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_builder.preprocess.audio_features import extract_audio_embedding
from dataset_builder.preprocess.face_identity import ArcFaceIdentityEncoder, infer_identity_hash


def _write_test_wave(path: Path, sample_rate: int = 16000, duration_sec: float = 0.25) -> None:
    timeline = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.25 * np.sin(2 * np.pi * 440.0 * timeline)
    pcm = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_audio_embedding_has_expected_size_and_metadata(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_test_wave(audio_path)
    embedding, metadata = extract_audio_embedding(audio_path)
    assert embedding.shape == (64,)
    assert metadata["sample_rate"] == 16000.0
    assert metadata["duration_sec"] > 0.2


def test_identity_encoder_is_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (64, 64), color=(120, 80, 200)).save(image_path)
    encoder = ArcFaceIdentityEncoder()
    first = encoder.encode(str(image_path))
    second = encoder.encode(str(image_path))
    assert encoder.backend == "deterministic_fallback"
    assert first == second
    assert len(first) > 40
    assert infer_identity_hash(first) == infer_identity_hash(second)
