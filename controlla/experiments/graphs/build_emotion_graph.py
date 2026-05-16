"""Build the emotion graph used by Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np

from .graph_utils import infer_graph_output_dir, load_manifest, save_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Controlla emotion graph")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_main.yaml")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, manifest, _ = load_manifest(args.config, args.manifest_path)
    num_samples = len(manifest)
    adjacency = np.zeros((num_samples, num_samples), dtype=np.float32)
    emotions = manifest["unified_emotion"].fillna("neutral").astype(str).tolist()
    valence = manifest["valence"].fillna(0.0).to_numpy(dtype=np.float32) if "valence" in manifest.columns else np.zeros(num_samples, dtype=np.float32)
    arousal = manifest["arousal"].fillna(0.0).to_numpy(dtype=np.float32) if "arousal" in manifest.columns else np.zeros(num_samples, dtype=np.float32)
    affect = np.stack([valence, arousal], axis=1)

    for index in range(num_samples):
        distances = np.linalg.norm(affect - affect[index], axis=1)
        candidates = np.argsort(distances)
        adjacency[index, index] = 1.0
        count = 0
        for neighbor in candidates:
            if neighbor == index:
                continue
            same_emotion = emotions[index] == emotions[neighbor]
            weight = 1.0 / (1.0 + distances[neighbor])
            if same_emotion:
                weight += 1.0
            adjacency[index, neighbor] = weight
            count += 1
            if count >= args.k:
                break
    adjacency = np.maximum(adjacency, adjacency.T)
    output_dir = infer_graph_output_dir(config, args.output_dir)
    save_graph(
        output_dir,
        "emotion_graph",
        adjacency,
        manifest,
        {
            "num_samples": num_samples,
            "k": args.k,
            "emotion_counts": manifest["unified_emotion"].fillna("neutral").value_counts().to_dict(),
        },
    )
    print(f"saved emotion graph to {output_dir}")


if __name__ == "__main__":
    main()