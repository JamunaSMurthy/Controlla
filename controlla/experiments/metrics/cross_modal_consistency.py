"""Cross-modal consistency metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import ImageBindAdapter

from .metric_utils import first_available_column, load_prediction_manifest, resolve_existing_paths


def compute_cross_modal_consistency(manifest_path: str) -> dict[str, float]:
    """Measure agreement between available text, image, and audio modalities."""
    frame = load_prediction_manifest(manifest_path)
    adapter = ImageBindAdapter()
    texts = first_available_column(frame, ["output_text", "generated_text", "text"]).astype(str).tolist()
    images = first_available_column(frame, ["output_image_path", "generated_image_path", "image_path"]).astype(str).tolist()
    audios = first_available_column(frame, ["output_audio_path", "generated_audio_path", "audio_path"]).astype(str).tolist()

    text_embeddings = adapter.encode_texts([text or f"missing-text::{index}" for index, text in enumerate(texts)])
    image_embeddings = adapter.encode_images(resolve_existing_paths(images, "consistency-image"))
    audio_embeddings = adapter.encode_audios(resolve_existing_paths(audios, "consistency-audio"))

    text_image = np.sum(text_embeddings * image_embeddings, axis=1)
    text_audio = np.sum(text_embeddings * audio_embeddings, axis=1)
    image_audio = np.sum(image_embeddings * audio_embeddings, axis=1)
    overall = (text_image + text_audio + image_audio) / 3.0
    return {
        "text_image_consistency": float(np.mean(text_image)),
        "text_audio_consistency": float(np.mean(text_audio)),
        "image_audio_consistency": float(np.mean(image_audio)),
        "cross_modal_consistency_score": float(np.mean(overall)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cross-modal consistency metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_cross_modal_consistency(args.manifest_path))