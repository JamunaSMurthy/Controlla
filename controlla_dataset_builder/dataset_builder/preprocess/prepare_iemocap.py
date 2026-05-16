"""Prepare IEMOCAP normalized index."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from dataset_builder.config import load_builder_paths, load_yaml_config
from dataset_builder.schemas import NormalizedRecord
from dataset_builder.utils.logging import get_logger

from .unify_labels import map_dataset_emotion, normalize_emotion


LOGGER = get_logger("dataset_builder.preprocess.iemocap")

EVAL_LINE_RE = re.compile(
    r"^\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s+"
    r"(?P<turn>\S+)\s+(?P<label>\w+)\s+"
    r"\[(?P<valence>[0-9.]+),\s*(?P<arousal>[0-9.]+),\s*(?P<dominance>[0-9.]+)\]$"
)
TRANSCRIPT_RE = re.compile(r"^(?P<turn>\S+)\s+\[[^\]]+\]:\s*(?P<text>.*)$")


def parse_evaluation_line(line: str) -> dict[str, str] | None:
    match = EVAL_LINE_RE.match(line.strip())
    if not match:
        return None
    return match.groupdict()


def _load_transcript_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = TRANSCRIPT_RE.match(line.strip())
            if match:
                mapping[match.group("turn")] = match.group("text")
    return mapping


def _speaker_id(session_dir: Path, turn_name: str) -> str:
    return f"{session_dir.name}_{turn_name.rsplit('_', 1)[-1][0]}"


def build_records(raw_root: Path, dataset_settings: dict[str, object], raw_datasets_root: Path) -> list[NormalizedRecord]:
    del dataset_settings, raw_datasets_root
    records: list[NormalizedRecord] = []
    skipped_unresolved = 0
    for session_dir in sorted(path for path in raw_root.iterdir() if path.is_dir() and path.name.startswith("Session")):
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        transcript_dir = session_dir / "dialog" / "transcriptions"
        for eval_path in sorted(eval_dir.glob("*.txt")):
            dialog_name = eval_path.stem
            transcript_map = _load_transcript_map(transcript_dir / f"{dialog_name}.txt")
            with eval_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parsed = parse_evaluation_line(line)
                    if parsed is None:
                        continue
                    raw_label = parsed["label"].lower()
                    if raw_label == "xxx":
                        continue
                    mapped_label = map_dataset_emotion("iemocap", raw_label)
                    if mapped_label is None:
                        continue
                    turn_name = parsed["turn"]
                    audio_path = session_dir / "sentences" / "wav" / dialog_name / f"{turn_name}.wav"
                    if not audio_path.exists():
                        skipped_unresolved += 1
                        continue
                    text = transcript_map.get(turn_name)
                    records.append(
                        NormalizedRecord(
                            source_dataset="iemocap",
                            sample_id=turn_name,
                            audio_path=str(audio_path.resolve()),
                            text=text,
                            raw_text=text,
                            raw_label=raw_label,
                            unified_emotion=mapped_label,
                            identity_id=_speaker_id(session_dir, turn_name),
                            speaker_id=_speaker_id(session_dir, turn_name),
                            valence=float(parsed["valence"]),
                            arousal=float(parsed["arousal"]),
                        )
                    )
    if skipped_unresolved:
        LOGGER.warning("Skipped %s IEMOCAP utterances with missing audio", skipped_unresolved)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IEMOCAP normalized index")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    paths = load_builder_paths(args.config)
    config = load_yaml_config(args.config)
    dataset_settings = config["datasets"]["iemocap"]
    records = build_records(paths.raw_datasets_root / str(dataset_settings["raw_root"]), dataset_settings, paths.raw_datasets_root)
    LOGGER.info("Prepared %s IEMOCAP records", len(records))


if __name__ == "__main__":
    main()