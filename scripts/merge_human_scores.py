"""
Merge a completed blinded annotation sheet into results/metrics/human_scores.csv.

This script validates that every response has exactly one score in {1,2,3,4}
before writing the postprocess-compatible human score file.
"""

import argparse
import csv
import json
from pathlib import Path

from _common import CONDITION_ORDER, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-dir",
        default=None,
        help="Directory containing human_scoring_sheet.csv and human_scoring_key.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results/metrics/human_scores.csv.",
    )
    return parser.parse_args()


def load_pairs(cfg):
    with (Path(cfg["data_dir"]) / "processed" / "eval_pairs.json").open() as f:
        return json.load(f)


def read_key(path):
    key = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            annotation_id = int(row["annotation_id"])
            if annotation_id in key:
                raise ValueError(f"Duplicate annotation_id in key: {annotation_id}")
            key[annotation_id] = (int(row["pair_id"]), row["condition"])
    return key


def read_scores(path, key):
    scores = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            annotation_id = int(row["annotation_id"])
            if annotation_id not in key:
                raise ValueError(f"annotation_id={annotation_id} is missing from answer key")
            value = row["score"].strip()
            if not value:
                raise ValueError(f"Missing score for annotation_id={annotation_id}")
            try:
                score = int(value)
            except ValueError as exc:
                raise ValueError(f"Non-integer score for annotation_id={annotation_id}: {value}") from exc
            if score not in (1, 2, 3, 4):
                raise ValueError(f"Invalid score for annotation_id={annotation_id}: {score}")

            pair_id, condition = key[annotation_id]
            score_key = (pair_id, condition)
            if score_key in scores:
                raise ValueError(f"Duplicate score for pair_id={pair_id}, condition={condition}")
            scores[score_key] = score
    return scores


def main():
    args = parse_args()
    cfg = load_config()
    annotation_dir = (
        Path(args.annotation_dir) if args.annotation_dir else Path(cfg["output_dir"]) / "annotation"
    )
    sheet_path = annotation_dir / "human_scoring_sheet.csv"
    key_path = annotation_dir / "human_scoring_key.csv"
    output_path = Path(cfg["output_dir"]) / "metrics" / "human_scores.csv"

    if not sheet_path.exists():
        raise FileNotFoundError(f"Missing completed annotation sheet: {sheet_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"Missing annotation answer key: {key_path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing score file: {output_path}")

    pairs = load_pairs(cfg)
    expected = {
        (int(pair["pair_id"]), condition)
        for pair in pairs
        for condition in CONDITION_ORDER
    }
    key = read_key(key_path)
    scores = read_scores(sheet_path, key)

    if set(scores) != expected:
        missing = sorted(expected - set(scores))
        extra = sorted(set(scores) - expected)
        raise ValueError(
            f"Completed scores do not match the full design. Missing={len(missing)}, Extra={len(extra)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "condition", "score"])
        for pair in pairs:
            pair_id = int(pair["pair_id"])
            for condition in CONDITION_ORDER:
                writer.writerow([pair_id, condition, scores[(pair_id, condition)]])

    print(f"Wrote independent human scores to {output_path}")
    print("Next step: run bash run_postprocess.sh")


if __name__ == "__main__":
    main()
