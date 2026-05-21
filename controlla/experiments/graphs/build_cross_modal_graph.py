"""Build the cross-modal affinity graph used by Controlla experiments.

Paper alignment:
    The cross-modal graph captures multimodal affect/control affinity across
    image, text, and audio evidence. It is used for diagnostics and auxiliary
    cross-modal consistency experiments, not as a replacement for the emotion
    and identity graph priors.
"""

from __future__ import annotations

import argparse

import numpy as np

from controlla.evaluation.encoders import ImageBindAdapter

from .graph_utils import (
    cosine_similarity_matrix,
    get_column,
    infer_graph_output_dir,
    knn_adjacency,
    load_manifest,
    safe_string_series,
    save_graph,
)


def _mean_embedding(parts: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(parts, axis=0)
    return stacked.mean(axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Controlla cross-modal graph")
    parser.add_argument("--config", type=str, default="experiments/configs/datasets.yaml")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config, manifest, manifest_path = load_manifest(args.config, args.manifest_path)

    image_col = get_column(
        manifest,
        preferred="image_path",
        alternatives=["image", "target_image_path", "ximg"],
        required=False,
    )
    audio_col = get_column(
        manifest,
        preferred="audio_path",
        alternatives=["audio", "audio_feature_path", "xaud"],
        required=False,
    )
    text_col = get_column(
        manifest,
        preferred="prompt",
        alternatives=["text", "caption", "xtext"],
        required=False,
    )

    image_paths = safe_string_series(manifest, image_col, "missing-image")
    audio_paths = safe_string_series(manifest, audio_col, "missing-audio")
    texts = safe_string_series(manifest, text_col, "missing-text")

    adapter = ImageBindAdapter(device=args.device)

    image_embeddings = adapter.encode_images(image_paths)
    audio_embeddings = adapter.encode_audios(audio_paths)
    text_embeddings = adapter.encode_texts(texts)

    sample_embeddings = []

    for index in range(len(manifest)):
        parts = []

        if image_col is not None and str(manifest.iloc[index].get(image_col, "")).strip():
            parts.append(image_embeddings[index])

        if audio_col is not None and str(manifest.iloc[index].get(audio_col, "")).strip():
            parts.append(audio_embeddings[index])

        if text_col is not None and str(manifest.iloc[index].get(text_col, "")).strip():
            parts.append(text_embeddings[index])

        if not parts:
            parts.append(text_embeddings[index])

        sample_embeddings.append(_mean_embedding(parts))

    similarity = cosine_similarity_matrix(np.stack(sample_embeddings, axis=0))
    adjacency = knn_adjacency(similarity, k=args.k, self_loops=False)

    output_dir = infer_graph_output_dir(config, args.output_dir)

    save_graph(
        output_dir=output_dir,
        graph_name="cross_modal_graph",
        adjacency=adjacency,
        manifest=manifest,
        metadata={
            "manifest_path": str(manifest_path),
            "num_samples": int(len(manifest)),
            "k": int(args.k),
            "image_column": image_col,
            "audio_column": audio_col,
            "text_column": text_col,
            "encoder": "ImageBindAdapter",
            "builder": "build_cross_modal_graph.py",
        },
    )

    print(f"saved cross-modal graph to {output_dir}")


if __name__ == "__main__":
    main()