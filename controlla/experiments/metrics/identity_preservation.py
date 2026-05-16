"""Identity-preservation metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import ArcFaceAdapter

from .metric_utils import first_available_column, load_prediction_manifest, resolve_existing_paths


def compute_identity_preservation(manifest_path: str) -> dict[str, float]:
    """Measure how well generated samples preserve source identity."""
    frame = load_prediction_manifest(manifest_path)
    adapter = ArcFaceAdapter()
    reference_paths = first_available_column(frame, ["reference_image_path", "source_image_path", "image_path"]).astype(str).tolist()
    output_paths = first_available_column(frame, ["output_image_path", "generated_image_path", "image_path"]).astype(str).tolist()
    reference_embeddings = adapter.encode_images(resolve_existing_paths(reference_paths, "identity-reference"))
    output_embeddings = adapter.encode_images(resolve_existing_paths(output_paths, "identity-output"))
    similarity = np.sum(reference_embeddings * output_embeddings, axis=1)
    return {
        "identity_cosine_similarity": float(np.mean(similarity)),
        "identity_preservation_score": float(np.mean((similarity + 1.0) / 2.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute identity-preservation metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_identity_preservation(args.manifest_path))