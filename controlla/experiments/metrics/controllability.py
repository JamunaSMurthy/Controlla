"""Controllability metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import CLIPAdapter

from .metric_utils import (
    cosine_similarity,
    first_available_column,
    load_prediction_manifest,
    mean_or_nan,
    resolve_existing_paths,
    safe_float,
)


def _emotion_accuracy_values(frame) -> list[float | None]:
    """Compute emotion-control accuracy from available columns."""

    explicit_scores = first_available_column(frame, ["alignment_emotion_match"], "").map(safe_float).tolist()
    if any(value is not None for value in explicit_scores):
        return explicit_scores

    targets = first_available_column(
        frame,
        ["target_emotion", "unified_emotion", "emotion", "emotion_label"],
    ).astype(str).tolist()

    predictions = first_available_column(
        frame,
        ["predicted_emotion", "generated_emotion", "output_emotion"],
    ).astype(str).tolist()

    if any(value.strip() for value in predictions):
        return [
            float(target.lower() == pred.lower()) if target and pred else None
            for target, pred in zip(targets, predictions)
        ]

    generated_texts = first_available_column(
        frame,
        ["output_text", "generated_text", "text", "prompt"],
    ).astype(str).tolist()

    return [
        float(target.lower() in text.lower()) if target and text else None
        for target, text in zip(targets, generated_texts)
    ]


def compute_controllability(
    manifest_path: str,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure prompt-image alignment and explicit emotion control.

    Paper alignment:
        Acc should be computed from classifier/predicted_emotion when available.
        CLIP prompt-image alignment is auxiliary and should not replace Acc.
    """

    frame = load_prediction_manifest(manifest_path)
    adapter = CLIPAdapter(device=device)

    prompts = first_available_column(frame, ["target_prompt", "prompt", "text"], "").astype(str).tolist()
    emotions = first_available_column(frame, ["target_emotion", "unified_emotion", "emotion"], "").astype(str).tolist()

    final_prompts = [
        prompt if prompt.strip() else f"a {emotion} face"
        for prompt, emotion in zip(prompts, emotions)
    ]

    image_paths = first_available_column(
        frame,
        ["generated_image_path", "output_image_path", "prediction_path", "image_path"],
    ).astype(str).tolist()

    image_embeddings = adapter.encode_images(resolve_existing_paths(image_paths, "controllability-image"))
    prompt_embeddings = adapter.encode_texts(final_prompts)

    prompt_image_scores = cosine_similarity(prompt_embeddings, image_embeddings)
    emotion_acc = mean_or_nan(_emotion_accuracy_values(frame))

    if np.isnan(emotion_acc):
        controllability_score = float(np.mean(prompt_image_scores))
    else:
        controllability_score = float(0.5 * np.mean(prompt_image_scores) + 0.5 * emotion_acc)

    return {
        "Acc": emotion_acc,
        "prompt_image_alignment": float(np.mean(prompt_image_scores)),
        "emotion_control_accuracy": emotion_acc,
        "controllability_score": controllability_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute controllability metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_controllability(args.manifest_path, device=args.device))