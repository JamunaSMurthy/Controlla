"""Build the emotion / affective graph used by Controlla experiments.

Paper alignment:
    The emotion graph G_e captures relational structure among affective states.
    When valence/arousal columns are available, they are used. Otherwise, the
    graph is built from normalized emotion-label prototypes.
"""

from __future__ import annotations

import argparse

import numpy as np

from .graph_utils import (
    get_column,
    infer_graph_output_dir,
    knn_adjacency,
    load_manifest,
    save_graph,
)


DEFAULT_EMOTION_ORDER = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "contemptuous",
]


EMOTION_VALENCE_AROUSAL = {
    "neutral": (0.0, 0.0),
    "happy": (0.8, 0.5),
    "sad": (-0.7, -0.4),
    "angry": (-0.7, 0.7),
    "fearful": (-0.8, 0.8),
    "surprised": (0.2, 0.8),
    "disgusted": (-0.8, 0.3),
    "contemptuous": (-0.5, 0.2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Controlla emotion graph")
    parser.add_argument("--config", type=str, default="experiments/configs/datasets.yaml")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--node-level",
        type=str,
        default="sample",
        choices=["sample", "class"],
        help="Build sample-level or class-level emotion graph.",
    )
    return parser.parse_args()


def _emotion_to_va(emotion: str) -> tuple[float, float]:
    emotion = emotion.strip().lower()
    return EMOTION_VALENCE_AROUSAL.get(emotion, (0.0, 0.0))


def _build_sample_features(manifest) -> tuple[np.ndarray, dict]:
    emotion_col = get_column(
        manifest,
        preferred="emotion_label",
        alternatives=["unified_emotion", "emotion", "target_emotion"],
        required=False,
    )

    valence_col = get_column(manifest, "valence", required=False)
    arousal_col = get_column(manifest, "arousal", required=False)

    if valence_col is not None and arousal_col is not None:
        valence = manifest[valence_col].fillna(0.0).to_numpy(dtype=np.float32)
        arousal = manifest[arousal_col].fillna(0.0).to_numpy(dtype=np.float32)
        features = np.stack([valence, arousal], axis=1).astype(np.float32)
    elif emotion_col is not None:
        emotions = manifest[emotion_col].fillna("neutral").astype(str).tolist()
        features = np.asarray([_emotion_to_va(x) for x in emotions], dtype=np.float32)
    else:
        features = np.zeros((len(manifest), 2), dtype=np.float32)

    metadata = {
        "feature_type": "valence_arousal",
        "node_level": "sample",
        "num_samples": int(len(manifest)),
    }

    if emotion_col is not None:
        metadata["emotion_counts"] = (
            manifest[emotion_col].fillna("neutral").astype(str).value_counts().to_dict()
        )

    return features, metadata


def _build_class_features(manifest) -> tuple[np.ndarray, list[str], dict]:
    emotion_col = get_column(
        manifest,
        preferred="emotion_label",
        alternatives=["unified_emotion", "emotion", "target_emotion"],
        required=False,
    )

    if emotion_col is not None:
        observed = [
            str(x).strip().lower()
            for x in manifest[emotion_col].fillna("neutral").astype(str).unique().tolist()
        ]
        labels = [label for label in DEFAULT_EMOTION_ORDER if label in observed]
        labels.extend(sorted(set(observed) - set(labels)))
    else:
        labels = list(DEFAULT_EMOTION_ORDER)

    features = np.asarray([_emotion_to_va(label) for label in labels], dtype=np.float32)

    metadata = {
        "feature_type": "class_valence_arousal",
        "node_level": "class",
        "emotion_labels": labels,
        "num_classes": int(len(labels)),
    }

    return features, labels, metadata


def main() -> None:
    args = parse_args()

    config, manifest, manifest_path = load_manifest(args.config, args.manifest_path)

    if args.node_level == "class":
        features, labels, metadata = _build_class_features(manifest)
        similarity = 1.0 / (1.0 + np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1))
        adjacency = knn_adjacency(similarity, k=args.k, self_loops=False, nonnegative=False)

        # For class graph, create a tiny manifest-like table with sample_id labels.
        graph_manifest = manifest.iloc[: len(labels)].copy()
        if len(graph_manifest) < len(labels):
            import pandas as pd

            graph_manifest = pd.DataFrame({"sample_id": labels})
        else:
            graph_manifest = graph_manifest.copy()
            graph_manifest["sample_id"] = labels

    else:
        features, metadata = _build_sample_features(manifest)
        similarity = 1.0 / (1.0 + np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1))
        adjacency = knn_adjacency(similarity, k=args.k, self_loops=False, nonnegative=False)
        graph_manifest = manifest

    metadata.update(
        {
            "manifest_path": str(manifest_path),
            "k": int(args.k),
            "builder": "build_emotion_graph.py",
        }
    )

    output_dir = infer_graph_output_dir(config, args.output_dir)

    save_graph(
        output_dir=output_dir,
        graph_name="emotion_graph",
        adjacency=adjacency,
        manifest=graph_manifest,
        metadata=metadata,
    )

    print(f"saved emotion graph to {output_dir}")


if __name__ == "__main__":
    main()