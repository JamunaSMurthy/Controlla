"""Build the cross-modal affinity graph used by Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import ImageBindAdapter

from .graph_utils import cosine_similarity_matrix, infer_graph_output_dir, knn_adjacency, load_manifest, save_graph


def _mean_embedding(parts: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(parts, axis=0)
    return stacked.mean(axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Controlla cross-modal graph")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_main.yaml")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, manifest, _ = load_manifest(args.config, args.manifest_path)
    adapter = ImageBindAdapter()
    image_paths = manifest["image_path"].fillna("").astype(str).tolist()
    audio_paths = manifest["audio_path"].fillna("").astype(str).tolist()
    texts = manifest["text"].fillna("").astype(str).tolist()
    image_embeddings = adapter.encode_images([path or f"missing-image::{index}" for index, path in enumerate(image_paths)])
    audio_embeddings = adapter.encode_audios([path or f"missing-audio::{index}" for index, path in enumerate(audio_paths)])
    text_embeddings = adapter.encode_texts([text or f"missing-text::{index}" for index, text in enumerate(texts)])

    sample_embeddings = []
    for index in range(len(manifest)):
        parts = []
        if image_paths[index]:
            parts.append(image_embeddings[index])
        if audio_paths[index]:
            parts.append(audio_embeddings[index])
        if texts[index]:
            parts.append(text_embeddings[index])
        if not parts:
            parts.append(text_embeddings[index])
        sample_embeddings.append(_mean_embedding(parts))
    similarity = cosine_similarity_matrix(np.stack(sample_embeddings, axis=0))
    adjacency = knn_adjacency(similarity, args.k)

    output_dir = infer_graph_output_dir(config, args.output_dir)
    save_graph(
        output_dir,
        "cross_modal_graph",
        adjacency,
        manifest,
        {"num_samples": len(manifest), "k": args.k},
    )
    print(f"saved cross-modal graph to {output_dir}")


if __name__ == "__main__":
    main()