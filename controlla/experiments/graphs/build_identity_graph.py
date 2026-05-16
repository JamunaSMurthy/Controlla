"""Build the identity graph used by Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import ArcFaceAdapter

from .graph_utils import cosine_similarity_matrix, infer_graph_output_dir, knn_adjacency, load_manifest, save_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Controlla identity graph")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_main.yaml")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, manifest, _ = load_manifest(args.config, args.manifest_path)
    image_paths = manifest["image_path"].fillna("").astype(str).tolist()
    adapter = ArcFaceAdapter()
    embeddings = adapter.encode_images([path or f"missing::{index}" for index, path in enumerate(image_paths)])
    similarity = cosine_similarity_matrix(embeddings)
    adjacency = knn_adjacency(similarity, args.k)

    identities = manifest["identity_id"].fillna("").astype(str).tolist()
    for source in range(len(identities)):
        for target in range(len(identities)):
            if identities[source] and identities[source] == identities[target]:
                adjacency[source, target] = max(adjacency[source, target], 1.0)

    output_dir = infer_graph_output_dir(config, args.output_dir)
    save_graph(
        output_dir,
        "identity_graph",
        adjacency,
        manifest,
        {
            "num_samples": len(manifest),
            "k": args.k,
            "num_known_identities": int((manifest["identity_id"].fillna("") != "").sum()),
        },
    )
    print(f"saved identity graph to {output_dir}")


if __name__ == "__main__":
    main()