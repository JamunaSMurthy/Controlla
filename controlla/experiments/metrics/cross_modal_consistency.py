"""Cross-modal consistency metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import ImageBindAdapter

from .metric_utils import (
    cosine_similarity,
    first_available_column,
    load_prediction_manifest,
    resolve_existing_paths,
    stable_recall_at_k,
)


def compute_cross_modal_consistency(
    manifest_path: str,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure agreement between text, image, and audio modalities.

    Paper alignment:
        ImageBind-style image-text and image-audio consistency contributes to
        the IB / cross-modal consistency diagnostics.
    """

    frame = load_prediction_manifest(manifest_path)
    adapter = ImageBindAdapter(device=device)

    emotions = first_available_column(frame, ["target_emotion", "unified_emotion", "emotion"], "").astype(str).tolist()
    texts = first_available_column(frame, ["target_prompt", "prompt", "text"], "").astype(str).tolist()
    texts = [text if text.strip() else f"a {emotion} face" for text, emotion in zip(texts, emotions)]

    image_paths = first_available_column(
        frame,
        ["generated_image_path", "output_image_path", "prediction_path", "image_path"],
    ).astype(str).tolist()

    audio_paths = first_available_column(
        frame,
        ["generated_audio_path", "output_audio_path", "audio_path", "audio_feature_path"],
    ).astype(str).tolist()

    text_embeddings = adapter.encode_texts(texts)
    image_embeddings = adapter.encode_images(resolve_existing_paths(image_paths, "consistency-image"))
    audio_embeddings = adapter.encode_audios(resolve_existing_paths(audio_paths, "consistency-audio"))

    text_image = cosine_similarity(text_embeddings, image_embeddings)
    text_audio = cosine_similarity(text_embeddings, audio_embeddings)
    image_audio = cosine_similarity(image_embeddings, audio_embeddings)

    ib_score = 0.5 * image_audio + 0.5 * text_image

    retrieval = {}
    retrieval.update(stable_recall_at_k(image_embeddings, text_embeddings, prefix="I2T"))
    retrieval.update(stable_recall_at_k(image_embeddings, audio_embeddings, prefix="I2A"))

    return {
        "IB": float(np.mean(ib_score)),
        "text_image_consistency": float(np.mean(text_image)),
        "text_audio_consistency": float(np.mean(text_audio)),
        "image_audio_consistency": float(np.mean(image_audio)),
        "cross_modal_consistency_score": float(np.mean((text_image + text_audio + image_audio) / 3.0)),
        **retrieval,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cross-modal consistency metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_cross_modal_consistency(args.manifest_path, device=args.device))