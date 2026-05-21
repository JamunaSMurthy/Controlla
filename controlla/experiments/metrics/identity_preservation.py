"""Identity-preservation metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import ArcFaceAdapter

from .metric_utils import (
    cosine_similarity,
    first_available_column,
    load_prediction_manifest,
    resolve_existing_paths,
)


def compute_identity_preservation(
    manifest_path: str,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure generated-reference identity similarity.

    Paper alignment:
        This corresponds to ID / ID-Sim style identity preservation. The metric
        should be computed with real frozen ArcFace/identity features for paper
        reporting. Hash fallback is only for smoke tests.
    """

    frame = load_prediction_manifest(manifest_path)

    reference_paths = first_available_column(
        frame,
        ["reference_image_path", "source_image_path", "ref_image_path", "image_path"],
    ).astype(str).tolist()

    output_paths = first_available_column(
        frame,
        ["generated_image_path", "output_image_path", "prediction_path", "image_path"],
    ).astype(str).tolist()

    adapter = ArcFaceAdapter(device=device)

    reference_embeddings = adapter.encode_images(
        resolve_existing_paths(reference_paths, "identity-reference")
    )
    output_embeddings = adapter.encode_images(
        resolve_existing_paths(output_paths, "identity-output")
    )

    similarity = cosine_similarity(reference_embeddings, output_embeddings)
    identity_score = (similarity + 1.0) / 2.0

    return {
        "ID": float(np.mean(identity_score)),
        "identity_cosine_similarity": float(np.mean(similarity)),
        "identity_preservation_score": float(np.mean(identity_score)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute identity-preservation metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_identity_preservation(args.manifest_path, device=args.device))