"""Controllability metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import CLIPAdapter

from .metric_utils import first_available_column, load_prediction_manifest, resolve_existing_paths


def compute_controllability(manifest_path: str) -> dict[str, float]:
    """Measure prompt-image alignment and explicit emotion control accuracy."""
    frame = load_prediction_manifest(manifest_path)
    adapter = CLIPAdapter()
    prompts = first_available_column(frame, ["target_prompt", "prompt", "text"]).astype(str).tolist()
    emotions = first_available_column(frame, ["target_emotion", "unified_emotion"]).astype(str).tolist()
    image_paths = first_available_column(frame, ["output_image_path", "generated_image_path", "image_path"]).astype(str).tolist()
    image_embeddings = adapter.encode_images(resolve_existing_paths(image_paths, "controllability-image"))
    prompt_embeddings = adapter.encode_texts([prompt if prompt else f"a {emotion} portrait" for prompt, emotion in zip(prompts, emotions)])
    prompt_image_scores = np.sum(prompt_embeddings * image_embeddings, axis=1)

    predicted_texts = first_available_column(frame, ["output_text", "generated_text", "text"]).astype(str).tolist()
    emotion_matches = []
    for target, predicted in zip(emotions, predicted_texts):
        if not target:
            emotion_matches.append(0.0)
            continue
        emotion_matches.append(float(target.lower() in predicted.lower()) if predicted else 1.0)

    return {
        "prompt_image_alignment": float(np.mean(prompt_image_scores)),
        "emotion_control_accuracy": float(np.mean(emotion_matches)),
        "controllability_score": float(0.5 * np.mean(prompt_image_scores) + 0.5 * np.mean(emotion_matches)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute controllability metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_controllability(args.manifest_path))