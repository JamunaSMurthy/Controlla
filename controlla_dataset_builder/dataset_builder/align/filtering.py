"""Filtering and balancing over aligned tuples."""

from __future__ import annotations

from dataset_builder.schemas import AlignedSample


def filter_tuples(
    tuples: list[AlignedSample],
    *,
    minimum_final_score: float,
    require_audio: bool,
    require_text: bool,
    require_reference_image: bool,
    max_samples_per_class: int | None = None,
) -> list[AlignedSample]:
    filtered: list[AlignedSample] = []
    class_counts: dict[str, int] = {}
    for sample in sorted(tuples, key=lambda item: item.alignment_scores.final_score, reverse=True):
        if sample.alignment_scores.final_score < minimum_final_score:
            continue
        if require_audio and not sample.modality_mask.has_audio:
            continue
        if require_text and not sample.modality_mask.has_text:
            continue
        if require_reference_image and not sample.modality_mask.has_reference_image:
            continue
        if max_samples_per_class is not None and class_counts.get(sample.unified_emotion, 0) >= max_samples_per_class:
            continue
        filtered.append(sample)
        class_counts[sample.unified_emotion] = class_counts.get(sample.unified_emotion, 0) + 1
    return filtered