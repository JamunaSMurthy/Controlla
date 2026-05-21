"""Build the reference-identity graph used by Controlla experiments.

Paper alignment:
    Identity in Controlla is grounded through the visual reference image x_ref.
    This graph therefore uses reference_image_path by default, falling back to
    image_path only when no reference column exists.
"""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import ArcFaceAdapter

from .graph_utils import (
    cosine_similarity_matrix,
    get_column,
    infer_graph_output_dir,
    knn_adjacency,
    load_manifest,
    safe_string_series,
    save_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Controlla identity graph")
    parser.add_argument("--config", type=str, default="experiments/configs/datasets.yaml")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config, manifest, manifest_path = load_manifest(args.config, args.manifest_path)

    reference_col = get_column(
        manifest,
        preferred="reference_image_path",
        alternatives=["reference_path", "xref", "image_path"],
        required=True,
    )

    identity_col = get_column(
        manifest,
        preferred="identity_id",
        alternatives=["reference_identity", "ref_identity", "person_id"],
        required=False,
    )

    reference_paths = safe_string_series(
        manifest=manifest,
        column=reference_col,
        default_prefix="missing-reference",
    )

    adapter = ArcFaceAdapter(device=args.device)
    embeddings = adapter.encode_images(reference_paths)

    similarity = cosine_similarity_matrix(embeddings)
    adjacency = knn_adjacency(similarity, k=args.k, self_loops=False)

    if identity_col is not None:
        identities = manifest[identity_col].fillna("").astype(str).tolist()

        for source in range(len(identities)):
            for target in range(len(identities)):
                if identities[source] and identities[source] == identities[target]:
                    adjacency[source, target] = max(float(adjacency[source, target]), 1.0)

        np.fill_diagonal(adjacency, 0.0)

    output_dir = infer_graph_output_dir(config, args.output_dir)

    save_graph(
        output_dir=output_dir,
        graph_name="identity_graph",
        adjacency=adjacency,
        manifest=manifest,
        metadata={
            "manifest_path": str(manifest_path),
            "num_samples": int(len(manifest)),
            "k": int(args.k),
            "reference_column": reference_col,
            "identity_column": identity_col,
            "num_known_identities": int(
                (manifest[identity_col].fillna("") != "").sum()
            )
            if identity_col is not None
            else 0,
            "encoder": "ArcFaceAdapter",
            "builder": "build_identity_graph.py",
        },
    )

    print(f"saved identity graph to {output_dir}")


if __name__ == "__main__":
    main()