"""Disentanglement metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from controlla.evaluation.encoders import ArcFaceAdapter, CLIPAdapter

from .metric_utils import (
    cosine_similarity,
    first_available_column,
    load_prediction_manifest,
    resolve_existing_paths,
    stack_vector_column,
)


def _safe_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if embeddings.shape[0] < 4 or len(np.unique(labels)) < 2:
        return float("nan")

    try:
        return float(max(0.0, silhouette_score(embeddings, labels, metric="cosine")))
    except Exception:
        return float("nan")


def _probe_accuracy(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    if features.shape[0] < 10 or len(np.unique(labels)) < 2:
        return float("nan")

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_labels = {label for label, count in zip(unique_labels, counts) if count >= 2}
    mask = np.asarray([label in valid_labels for label in labels], dtype=bool)

    filtered_features = features[mask]
    filtered_labels = labels[mask]

    if filtered_features.shape[0] < 10 or len(np.unique(filtered_labels)) < 2:
        return float("nan")

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            filtered_features,
            filtered_labels,
            test_size=0.25,
            random_state=seed,
            stratify=filtered_labels,
        )
        classifier = LogisticRegression(max_iter=500, random_state=seed)
        classifier.fit(x_train, y_train)
        return float(accuracy_score(y_test, classifier.predict(x_test)))
    except Exception:
        return float("nan")


def latent_disentanglement_score(z_attr: np.ndarray, z_id: np.ndarray) -> float:
    """Compute LDS from learned attribute and identity latents.

    Higher is better. This estimates leakage as absolute cosine overlap.
    """

    if z_attr.ndim != 2 or z_id.ndim != 2:
        raise ValueError("z_attr and z_id must have shape [N, D]")

    dim = min(z_attr.shape[1], z_id.shape[1])
    if dim == 0:
        return float("nan")

    overlap = np.abs(cosine_similarity(z_attr[:, :dim], z_id[:, :dim]))
    return float(np.clip(1.0 - np.mean(overlap), 0.0, 1.0))


def compute_disentanglement(
    manifest_path: str,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, float]:
    """Estimate identity/attribute separation.

    Paper alignment:
        If z_attr and z_id columns are present, LDS is computed directly from
        Controlla's learned factors. Otherwise, this falls back to external
        feature proxies for smoke-test evaluation.
    """

    frame = load_prediction_manifest(manifest_path)

    z_attr = stack_vector_column(frame, ["z_attr", "attribute_latent", "attr_latent"])
    z_id = stack_vector_column(frame, ["z_id", "identity_latent", "id_latent"])

    if z_attr is not None and z_id is not None:
        lds = latent_disentanglement_score(z_attr, z_id)
        return {
            "LDS": lds,
            "latent_disentanglement_score": lds,
            "identity_silhouette": float("nan"),
            "emotion_silhouette": float("nan"),
            "emotion_from_identity_leakage": float("nan"),
            "identity_from_attribute_leakage": float("nan"),
        }

    identity_col = "identity_id" if "identity_id" in frame.columns else None
    emotion_col = "target_emotion" if "target_emotion" in frame.columns else "unified_emotion" if "unified_emotion" in frame.columns else None

    if identity_col is None or emotion_col is None:
        return {
            "LDS": float("nan"),
            "latent_disentanglement_score": float("nan"),
            "identity_silhouette": float("nan"),
            "emotion_silhouette": float("nan"),
            "emotion_from_identity_leakage": float("nan"),
            "identity_from_attribute_leakage": float("nan"),
        }

    filtered = frame[
        (frame[identity_col].fillna("").astype(str) != "")
        & (frame[emotion_col].fillna("").astype(str) != "")
    ].copy()

    if filtered.empty:
        return {
            "LDS": float("nan"),
            "latent_disentanglement_score": float("nan"),
            "identity_silhouette": float("nan"),
            "emotion_silhouette": float("nan"),
            "emotion_from_identity_leakage": float("nan"),
            "identity_from_attribute_leakage": float("nan"),
        }

    identity_encoder = ArcFaceAdapter(device=device)
    attribute_encoder = CLIPAdapter(device=device)

    reference_paths = first_available_column(
        filtered,
        ["reference_image_path", "source_image_path", "image_path"],
    ).astype(str).tolist()

    identity_embeddings = identity_encoder.encode_images(
        resolve_existing_paths(reference_paths, "disentangle-id")
    )

    attribute_prompts = [f"{emotion}" for emotion in filtered[emotion_col].astype(str).tolist()]
    attribute_embeddings = attribute_encoder.encode_texts(attribute_prompts)

    identity_labels = LabelEncoder().fit_transform(filtered[identity_col].astype(str).tolist())
    emotion_labels = LabelEncoder().fit_transform(filtered[emotion_col].astype(str).tolist())

    identity_silhouette = _safe_silhouette(identity_embeddings, identity_labels)
    emotion_silhouette = _safe_silhouette(attribute_embeddings, emotion_labels)
    emotion_from_identity = _probe_accuracy(identity_embeddings, emotion_labels, seed)
    identity_from_attribute = _probe_accuracy(attribute_embeddings, identity_labels, seed)

    lds = latent_disentanglement_score(attribute_embeddings, identity_embeddings)

    return {
        "LDS": lds,
        "latent_disentanglement_score": lds,
        "identity_silhouette": identity_silhouette,
        "emotion_silhouette": emotion_silhouette,
        "emotion_from_identity_leakage": emotion_from_identity,
        "identity_from_attribute_leakage": identity_from_attribute,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute disentanglement metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_disentanglement(args.manifest_path, seed=args.seed, device=args.device))