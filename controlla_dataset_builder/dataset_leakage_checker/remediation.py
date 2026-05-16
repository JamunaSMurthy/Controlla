"""Remediation planning and split-fix application for leakage findings."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLIT_PRIORITY: dict[str, int] = {"train": 0, "val": 1, "test": 2}


def _sample_index(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {sample["sample_id"]: sample for sample in samples}


def _choose_target_split(group_samples: list[dict[str, Any]]) -> str:
    counts = Counter(sample["split"] for sample in group_samples)
    return sorted(counts.items(), key=lambda item: (-item[1], SPLIT_PRIORITY.get(item[0], 99), item[0]))[0][0]


def _build_move(group_name: str, key: str, group_samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    splits = sorted({sample["split"] for sample in group_samples})
    if len(splits) < 2:
        return None
    target_split = _choose_target_split(group_samples)
    moves = []
    for sample in sorted(group_samples, key=lambda item: (item["split"], item["sample_id"])):
        if sample["split"] == target_split:
            continue
        moves.append(
            {
                "sample_id": sample["sample_id"],
                "from_split": sample["split"],
                "to_split": target_split,
            }
        )
    if not moves:
        return None
    return {
        "group_type": group_name,
        "group_key": key,
        "target_split": target_split,
        "splits": splits,
        "sample_ids": [sample["sample_id"] for sample in group_samples],
        "moves": moves,
    }


def _groups_from_explicit_leaks(samples_by_id: dict[str, dict[str, Any]], leak_items: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for item in leak_items:
        group_samples = [samples_by_id[sample_id] for sample_id in item.get("sample_ids", []) if sample_id in samples_by_id]
        group = _build_move(key_name, str(item.get(key_name) or item.get("cluster_id") or "unknown"), group_samples)
        if group is not None:
            groups.append(group)
    return groups


def build_remediation_plan(report: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    samples_by_id = _sample_index(samples)
    grouped_moves: list[dict[str, Any]] = []
    grouped_moves.extend(
        _groups_from_explicit_leaks(
            samples_by_id,
            report["identity_leakage"]["explicit_identity_id"].get("leaked_identities", []),
            "identity_id",
        )
    )
    grouped_moves.extend(
        _groups_from_explicit_leaks(
            samples_by_id,
            report["speaker_leakage"]["explicit_speaker_id"].get("leaked_speakers", []),
            "speaker_id",
        )
    )
    grouped_moves.extend(
        _groups_from_explicit_leaks(
            samples_by_id,
            report["cross_modal_leakage"].get("identity_emotion_overlap", {}).get("items", []),
            "identity_id",
        )
    )
    grouped_moves.extend(
        _groups_from_explicit_leaks(
            samples_by_id,
            report["cross_modal_leakage"].get("speaker_emotion_overlap", {}).get("items", []),
            "speaker_id",
        )
    )

    deduped_moves: dict[str, dict[str, Any]] = {}
    reasons_by_sample: dict[str, set[str]] = defaultdict(set)
    for group in grouped_moves:
        for move in group["moves"]:
            sample_id = move["sample_id"]
            reasons_by_sample[sample_id].add(f"{group['group_type']}:{group['group_key']}")
            existing = deduped_moves.get(sample_id)
            if existing is None:
                deduped_moves[sample_id] = dict(move)
                continue
            if SPLIT_PRIORITY.get(move["to_split"], 99) < SPLIT_PRIORITY.get(existing["to_split"], 99):
                deduped_moves[sample_id] = dict(move)

    final_moves = []
    moved_counts = Counter()
    for sample_id, move in sorted(deduped_moves.items()):
        sample = samples_by_id[sample_id]
        final_moves.append(
            {
                **move,
                "reasons": sorted(reasons_by_sample[sample_id]),
                "dataset_source": sample.get("dataset_source"),
                "identity_id": sample.get("identity_id"),
                "speaker_id": sample.get("speaker_id"),
            }
        )
        moved_counts[move["from_split"]] += 1

    projected_counts = Counter(sample["split"] for sample in samples)
    for move in final_moves:
        projected_counts[move["from_split"]] -= 1
        projected_counts[move["to_split"]] += 1

    return {
        "num_samples": len(samples),
        "num_proposed_moves": len(final_moves),
        "moves": final_moves,
        "projected_split_counts": dict(projected_counts),
        "moved_from_counts": dict(moved_counts),
        "examples": final_moves[:20],
    }


def apply_remediation_plan(samples: list[dict[str, Any]], remediation_plan: dict[str, Any]) -> list[dict[str, Any]]:
    moves = {move["sample_id"]: move for move in remediation_plan.get("moves", [])}
    remediated = []
    for sample in samples:
        move = moves.get(sample["sample_id"])
        updated = dict(sample)
        if move is not None:
            updated["split"] = move["to_split"]
            updated["remediation_applied"] = True
            updated["remediation_reasons"] = move["reasons"]
        remediated.append(updated)
    return remediated


def _serializable_sample(sample: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in sample.items():
        if key.startswith("_"):
            continue
        if hasattr(value, "tolist"):
            value = value.tolist()
        serialized[key] = value
    return serialized


def write_remediation_outputs(
    output_dir: str | Path,
    remediation_plan: dict[str, Any],
    remediated_samples: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    plan_json = target_dir / "remediation_plan.json"
    plan_txt = target_dir / "remediation_plan.txt"
    plan_json.write_text(json.dumps(remediation_plan, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "Controlla Leakage Remediation Plan",
        "=",
        "",
        f"Proposed moves: {remediation_plan['num_proposed_moves']}",
        f"Projected split counts: {json.dumps(remediation_plan['projected_split_counts'], sort_keys=True)}",
        "",
        "Example moves:",
        json.dumps(remediation_plan["examples"], indent=2),
    ]
    plan_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {"plan_json": str(plan_json), "plan_txt": str(plan_txt)}
    if remediated_samples is not None:
        remediated_jsonl = target_dir / "remediated_dataset.jsonl"
        with remediated_jsonl.open("w", encoding="utf-8") as handle:
            for sample in remediated_samples:
                handle.write(json.dumps(_serializable_sample(sample), ensure_ascii=True) + "\n")
        outputs["remediated_jsonl"] = str(remediated_jsonl)
    return outputs