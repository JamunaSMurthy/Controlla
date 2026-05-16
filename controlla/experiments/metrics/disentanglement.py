"""Disentanglement metrics for Controlla experiments."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from controlla.evaluation.encoders import ArcFaceAdapter, CLIPAdapter

from .metric_utils import first_available_column, load_prediction_manifest, resolve_existing_paths


def _safe_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        return 0.0
    return float(max(0.0, silhouette_score(embeddings, labels, metric="cosine")))


def _probe_accuracy(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    if len(np.unique(labels)) < 2 or len(labels) < 10:
        return 0.0
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_labels = {label for label, count in zip(unique_labels, counts) if count >= 2}
    mask = np.asarray([label in valid_labels for label in labels], dtype=bool)
    filtered_features = features[mask]
    filtered_labels = labels[mask]
    if len(np.unique(filtered_labels)) < 2 or len(filtered_labels) < 10:
        return 0.0
    if len(np.unique(filtered_labels)) > max(2, len(filtered_labels) // 4):
        return 0.0
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


def compute_disentanglement(manifest_path: str, seed: int = 0) -> dict[str, float]:
    """Estimate identity/attribute separation using lightweight probing metrics."""
    frame = load_prediction_manifest(manifest_path)
    filtered = frame[(frame["identity_id"].fillna("") != "") & (frame["unified_emotion"].fillna("") != "")].copy()
    if filtered.empty:
        return {
            "identity_silhouette": 0.0,
            "emotion_silhouette": 0.0,
            "emotion_from_identity_leakage": 0.0,
            "identity_from_attribute_leakage": 0.0,
            "disentanglement_score": 0.0,
        }

    identity_encoder = ArcFaceAdapter()
    attribute_encoder = CLIPAdapter()
    image_paths = first_available_column(filtered, ["reference_image_path", "source_image_path", "image_path"]).astype(str).tolist()
    identity_embeddings = identity_encoder.encode_images(resolve_existing_paths(image_paths, "disentangle-id"))
    attribute_prompts = [f"{emotion}" for emotion in filtered["unified_emotion"].astype(str).tolist()]
    attribute_embeddings = attribute_encoder.encode_texts(attribute_prompts)

    identity_labels = LabelEncoder().fit_transform(filtered["identity_id"].astype(str).tolist())
    emotion_labels = LabelEncoder().fit_transform(filtered["unified_emotion"].astype(str).tolist())
    identity_silhouette = _safe_silhouette(identity_embeddings, identity_labels)
    emotion_silhouette = _safe_silhouette(attribute_embeddings, emotion_labels)
    emotion_from_identity = _probe_accuracy(identity_embeddings, emotion_labels, seed)
    identity_from_attribute = _probe_accuracy(attribute_embeddings, identity_labels, seed)
    score = max(0.0, 0.5 * (identity_silhouette + emotion_silhouette) - 0.5 * (emotion_from_identity + identity_from_attribute))
    return {
        "identity_silhouette": identity_silhouette,
        "emotion_silhouette": emotion_silhouette,
        "emotion_from_identity_leakage": emotion_from_identity,
        "identity_from_attribute_leakage": identity_from_attribute,
        "disentanglement_score": score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute disentanglement metrics")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(compute_disentanglement(args.manifest_path, seed=args.seed))