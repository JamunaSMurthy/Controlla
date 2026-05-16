"""Run the paper-level Controlla experiment suite.

This runner maps the result sections used in the paper to concrete, reusable
prediction-manifest evaluations. Controlla rows can run directly from the
AffectHuman manifest for smoke/release checks. External baselines require a
`predictions.csv` file in the configured prediction root and are otherwise
reported as missing instead of being silently faked.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from controlla.experiments.metrics.paper_metrics import (
    compute_core_paper_metrics,
    compute_retrieval_metrics,
    prepare_prediction_frame,
)
from controlla.utils import ensure_directory, load_config, resolve_project_path, write_json


DISPLAY_NAMES = {
    "styleclip": "StyleCLIP",
    "controlnet": "ControlNet",
    "controlnetpp": "ControlNet++",
    "icedit": "ICEdit",
    "flux": "FLUX",
    "flux_kontext": "FLUX.1 Kontext",
    "dreambooth": "DreamBooth",
    "dreambooth_controlnetpp": "DBooth + CNet++",
    "sdxl": "SDXL",
    "sdxl_controlnetpp": "SDXL + ControlNet++",
    "clip": "CLIP",
    "imagebind": "ImageBind",
    "controlla": "Controlla",
    "controlla_full": "Controlla",
    "contrastive_only": "Contrastive-only",
    "metric_no_ot": "Metric (no OT)",
    "without_emotion_fgw": "w/o FGW (Attr)",
    "without_identity_gw": "w/o GW (Id)",
    "without_disentanglement_loss": "w/o L_perp",
    "without_graph_euclidean": "w/o Graph (Euclidean)",
    "image_only": "Image only",
    "image_audio": "Image + Audio",
    "image_text": "Image + Text",
    "image_text_audio": "Image + Text + Audio",
    "linear_euclidean": "Linear (Euclidean)",
    "spline_interpolation": "Spline Interpolation",
    "controlla_geodesic": "Controlla (Geodesic)",
    "k_5": "k=5",
    "k_10": "k=10",
    "k_20": "k=20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-level Controlla experiments")
    parser.add_argument("--config", type=str, default="experiments/configs/eval_paper.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--prediction-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true", help="Do not include rows for missing external predictions.")
    return parser.parse_args()


def _display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _numeric_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _mean_std(rows: list[dict[str, Any]], metric: str) -> float:
    values = [_numeric_or_nan(row.get(metric)) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return float(pd.Series(values).mean()) if values else float("nan")


def _resolve_dataset_root(config: dict[str, Any], project_root: Path, override: str | None) -> Path:
    if override:
        root = resolve_project_path(project_root, override)
        if root is not None and root.exists():
            return root
        raise FileNotFoundError(f"Dataset root does not exist: {override}")

    configured = config.get("dataset_root")
    if configured:
        root = resolve_project_path(project_root, configured)
        if root is not None and root.exists():
            return root

    for candidate in config.get("dataset_root_candidates", []):
        root = resolve_project_path(project_root, candidate)
        if root is not None and (root / "metadata" / "dataset_manifest.csv").exists():
            return root
    raise FileNotFoundError("Could not resolve an AffectHuman dataset root from eval_paper.yaml")


def _resolve_manifest(config: dict[str, Any], project_root: Path, dataset_root: Path, override: str | None) -> Path:
    if override:
        manifest = resolve_project_path(project_root, override)
        if manifest is not None and manifest.exists():
            return manifest
        raise FileNotFoundError(f"Manifest does not exist: {override}")
    configured = config.get("manifest_path")
    if configured:
        manifest = resolve_project_path(project_root, configured)
        if manifest is not None and manifest.exists():
            return manifest
    manifest = dataset_root / "metadata" / "dataset_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest}")
    return manifest


def _prediction_candidates(prediction_root: Path, section: str, name: str, split: str, seed: int) -> list[Path]:
    return [
        prediction_root / section / name / split / f"seed_{seed}" / "predictions.csv",
        prediction_root / name / split / f"seed_{seed}" / "predictions.csv",
        prediction_root / section / name / f"seed_{seed}" / "predictions.csv",
        prediction_root / name / f"seed_{seed}" / "predictions.csv",
    ]


def _resolve_prediction(
    *,
    prediction_root: Path,
    section: str,
    name: str,
    split: str,
    seed: int,
    source_manifest: Path,
    fallback_names: set[str],
) -> tuple[Path | None, str]:
    for candidate in _prediction_candidates(prediction_root, section, name, split, seed):
        if candidate.exists():
            return candidate, "prediction_manifest"
    if name in fallback_names:
        return source_manifest, "source_manifest_fallback"
    return None, "missing_prediction_manifest"


def _evaluate_name(
    *,
    section: str,
    name: str,
    split: str,
    seed: int,
    source_manifest: Path,
    dataset_root: Path | None,
    prediction_root: Path,
    fallback_names: set[str],
    max_samples: int | None,
    metric_cache: dict[tuple[str, str, int | None, int, bool], dict[str, Any]],
    retrieval: bool = False,
) -> dict[str, Any]:
    prediction, status = _resolve_prediction(
        prediction_root=prediction_root,
        section=section,
        name=name,
        split=split,
        seed=seed,
        source_manifest=source_manifest,
        fallback_names=fallback_names,
    )
    row = {
        "name": name,
        "method": _display_name(name),
        "split": split,
        "seed": seed,
        "status": status,
        "prediction_manifest": str(prediction) if prediction is not None else "",
    }
    if prediction is None:
        return row
    cache_key = (str(prediction.resolve()), split, max_samples, seed, retrieval)
    if cache_key not in metric_cache:
        frame = prepare_prediction_frame(prediction, dataset_root=dataset_root, split=split, max_samples=max_samples)
        metrics = compute_retrieval_metrics(frame) if retrieval else compute_core_paper_metrics(frame, seed=seed)
        metric_cache[cache_key] = {**metrics, "num_samples": len(frame)}
    row.update(metric_cache[cache_key])
    return row


def _aggregate_seed_rows(rows: list[dict[str, Any]], id_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=id_columns + metric_columns)
    frame = pd.DataFrame.from_records(rows)
    grouped = frame.groupby(id_columns, dropna=False, sort=False)
    aggregated = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(id_columns, keys))
        statuses = sorted(set(group.get("status", pd.Series(dtype=str)).astype(str).tolist()))
        row["status"] = ",".join(statuses)
        row["num_samples"] = int(group["num_samples"].max()) if "num_samples" in group and group["num_samples"].notna().any() else 0
        for metric in metric_columns:
            row[metric] = _mean_std(group.to_dict("records"), metric)
        aggregated.append(row)
    return pd.DataFrame.from_records(aggregated)


def _write_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    ensure_directory(output_dir)
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, na_rep=""), encoding="utf-8")


def _run_split_method_table(
    *,
    section: str,
    names: list[str],
    splits: list[str],
    seeds: list[int],
    source_manifest: Path,
    dataset_root: Path,
    prediction_root: Path,
    fallback_names: set[str],
    max_samples: int | None,
    skip_missing: bool,
    metrics: list[str],
    metric_cache: dict[tuple[str, str, int | None, int, bool], dict[str, Any]],
    retrieval: bool = False,
) -> pd.DataFrame:
    rows = []
    for name in names:
        for split in splits:
            for seed in seeds:
                row = _evaluate_name(
                    section=section,
                    name=name,
                    split=split,
                    seed=seed,
                    source_manifest=source_manifest,
                    dataset_root=dataset_root,
                    prediction_root=prediction_root,
                    fallback_names=fallback_names,
                    max_samples=max_samples,
                    metric_cache=metric_cache,
                    retrieval=retrieval,
                )
                if skip_missing and row["status"] == "missing_prediction_manifest":
                    continue
                rows.append(row)
    return _aggregate_seed_rows(rows, ["name", "method", "split"], metrics)


def _wide_main_table(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for name, group in frame.groupby("name", sort=False):
        row = {"name": name, "method": group["method"].iloc[0], "status": ",".join(sorted(set(group["status"].astype(str))))}
        for split in ["val", "test"]:
            split_row = group[group["split"] == split]
            for metric in metrics:
                row[f"{split}_{metric}"] = split_row[metric].iloc[0] if not split_row.empty and metric in split_row else float("nan")
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _run_cross_dataset_table(
    *,
    config: dict[str, Any],
    project_root: Path,
    names: list[str],
    seeds: list[int],
    prediction_root: Path,
    fallback_names: set[str],
    max_samples: int | None,
    skip_missing: bool,
    metric_cache: dict[tuple[str, str, int | None, int, bool], dict[str, Any]],
) -> pd.DataFrame:
    manifests = config.get("cross_dataset_manifests", {})
    rows = []
    for name in names:
        for seed in seeds:
            output = {"name": name, "method": _display_name(name), "seed": seed, "status": "ok"}
            statuses = []
            for dataset_name, configured_manifest in manifests.items():
                manifest = resolve_project_path(project_root, configured_manifest)
                if manifest is None or not manifest.exists():
                    output[dataset_name] = float("nan")
                    statuses.append(f"missing_{dataset_name}_manifest")
                    continue
                prediction, status = _resolve_prediction(
                    prediction_root=prediction_root,
                    section=f"cross_dataset/{dataset_name}",
                    name=name,
                    split="test",
                    seed=seed,
                    source_manifest=manifest,
                    fallback_names=fallback_names,
                )
                statuses.append(status)
                if prediction is None:
                    output[dataset_name] = float("nan")
                    continue
                cache_key = (str(prediction.resolve()), "test", max_samples, seed, False)
                if cache_key not in metric_cache:
                    frame = prepare_prediction_frame(prediction, split="test", max_samples=max_samples)
                    metric_cache[cache_key] = {**compute_core_paper_metrics(frame, seed=seed), "num_samples": len(frame)}
                metrics = metric_cache[cache_key]
                output[dataset_name] = metrics["Acc"] if dataset_name == "affectnet" else metrics["ID"]
            output["status"] = ",".join(sorted(set(statuses)))
            if skip_missing and all(status.startswith("missing") for status in statuses):
                continue
            rows.append(output)
    return _aggregate_seed_rows(rows, ["name", "method"], list(manifests.keys()))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project_root = Path(config["project_root"])
    dataset_root = _resolve_dataset_root(config, project_root, args.dataset_root)
    source_manifest = _resolve_manifest(config, project_root, dataset_root, args.manifest_path)
    prediction_root = resolve_project_path(project_root, args.prediction_root or config["prediction_root"])
    output_root = resolve_project_path(project_root, args.output_root or config["output_root"])
    if prediction_root is None or output_root is None:
        raise ValueError("Could not resolve prediction/output roots")
    ensure_directory(prediction_root)
    ensure_directory(output_root)

    splits = [str(split) for split in config.get("splits", ["val", "test"])]
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    max_samples = args.max_samples if args.max_samples is not None else config.get("max_samples_per_split")
    fallback_names = {str(name) for name in config.get("controlla_fallback_methods", ["controlla"])}

    core_metrics = ["Acc", "TS", "CLIP", "IB", "H", "LDS", "GC", "ID"]
    retrieval_metrics = ["I2T_R@1", "I2T_R@5", "I2A_R@1", "I2A_R@5"]
    metric_cache: dict[tuple[str, str, int | None, int, bool], dict[str, Any]] = {}

    main_results = _run_split_method_table(
        section="main",
        names=config["main_methods"],
        splits=splits,
        seeds=seeds,
        source_manifest=source_manifest,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metrics=core_metrics,
        metric_cache=metric_cache,
    )
    _write_table(main_results, output_root, "main_results_long")
    _write_table(_wide_main_table(main_results, ["Acc", "TS", "IB", "H"]), output_root, "main_results")

    cross_dataset = _run_cross_dataset_table(
        config=config,
        project_root=project_root,
        names=config["cross_dataset_methods"],
        seeds=seeds,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metric_cache=metric_cache,
    )
    _write_table(cross_dataset, output_root, "cross_dataset_generalization")

    architecture = _run_split_method_table(
        section="architecture",
        names=config["architecture_methods"],
        splits=["val"],
        seeds=seeds,
        source_manifest=source_manifest,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metrics=["Acc", "TS", "LDS", "ID"],
        metric_cache=metric_cache,
    )
    _write_table(architecture, output_root, "architecture_comparison")

    geometry = _run_split_method_table(
        section="geometry",
        names=config["geometry_methods"],
        splits=splits,
        seeds=seeds,
        source_manifest=source_manifest,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metrics=["ID", "LDS", "GC"],
        metric_cache=metric_cache,
    )
    _write_table(_wide_main_table(geometry, ["ID", "LDS", "GC"]), output_root, "geometry_evaluation")

    retrieval = _run_split_method_table(
        section="retrieval",
        names=config["retrieval_methods"],
        splits=["test"],
        seeds=seeds,
        source_manifest=source_manifest,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metrics=retrieval_metrics,
        metric_cache=metric_cache,
        retrieval=True,
    )
    _write_table(retrieval, output_root, "cross_modal_retrieval")

    component = _run_split_method_table(
        section="ablations/components",
        names=config["component_ablations"],
        splits=["test"],
        seeds=seeds,
        source_manifest=source_manifest,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metrics=["Acc", "TS", "CLIP", "LDS", "GC", "ID"],
        metric_cache=metric_cache,
    )
    _write_table(component, output_root, "component_ablation")

    modality = _run_split_method_table(
        section="ablations/modalities",
        names=config["modality_ablations"],
        splits=["test"],
        seeds=seeds,
        source_manifest=source_manifest,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metrics=["Acc", "CLIP", "IB", "TS", "LDS", "H"],
        metric_cache=metric_cache,
    )
    _write_table(modality, output_root, "modality_ablation")

    traversal = _run_split_method_table(
        section="traversal",
        names=config["traversal_methods"],
        splits=["test"],
        seeds=seeds,
        source_manifest=source_manifest,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metrics=["TS", "LDS", "GC", "ID", "CLIP"],
        metric_cache=metric_cache,
    )
    _write_table(traversal, output_root, "traversal_comparison")

    sensitivity = _run_split_method_table(
        section="graph_sensitivity",
        names=config["graph_sensitivity"],
        splits=["test"],
        seeds=seeds,
        source_manifest=source_manifest,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        fallback_names=fallback_names,
        max_samples=max_samples,
        skip_missing=args.skip_missing,
        metrics=["TS", "LDS", "GC"],
        metric_cache=metric_cache,
    )
    _write_table(sensitivity, output_root, "graph_sensitivity")

    missing = []
    for table_name, frame in {
        "main_results": main_results,
        "cross_dataset_generalization": cross_dataset,
        "architecture_comparison": architecture,
        "geometry_evaluation": geometry,
        "cross_modal_retrieval": retrieval,
        "component_ablation": component,
        "modality_ablation": modality,
        "traversal_comparison": traversal,
        "graph_sensitivity": sensitivity,
    }.items():
        if "status" in frame:
            for _, row in frame[frame["status"].astype(str).str.contains("missing", na=False)].iterrows():
                missing.append({"table": table_name, "name": row.get("name", ""), "method": row.get("method", ""), "status": row.get("status", "")})
    pd.DataFrame.from_records(missing).drop_duplicates().to_csv(output_root / "missing_predictions.csv", index=False)
    write_json(
        {
            "dataset_root": str(dataset_root),
            "source_manifest": str(source_manifest),
            "prediction_root": str(prediction_root),
            "max_samples_per_split": max_samples,
            "splits": splits,
            "seeds": seeds,
        },
        output_root / "run_metadata.json",
    )
    print(f"wrote paper experiment tables to {output_root}")


if __name__ == "__main__":
    main()
