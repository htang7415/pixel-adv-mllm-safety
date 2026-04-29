"""
Prepare a blinded, randomized human-scoring sheet from model responses.

The scorer should fill only the score and notes columns. Condition labels are
kept in a separate answer key so scoring is not anchored on the attack type.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from _common import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for annotation files. Defaults to results/annotation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing annotation files.",
    )
    return parser.parse_args()


def load_pairs(cfg):
    with (Path(cfg["data_dir"]) / "processed" / "eval_pairs.json").open() as f:
        return json.load(f)


def load_responses(cfg):
    with (Path(cfg["output_dir"]) / "outputs" / "responses.json").open() as f:
        return json.load(f)


def main():
    args = parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])

    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"]) / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "human_scoring_sheet.csv"
    key_path = out_dir / "human_scoring_key.csv"

    if not args.force:
        existing = [path for path in (sheet_path, key_path) if path.exists()]
        if existing:
            paths = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing annotation files: {paths}")

    pairs = {int(pair["pair_id"]): pair for pair in load_pairs(cfg)}
    responses = load_responses(cfg)

    rows = []
    for response in responses:
        pair_id = int(response["pair_id"])
        pair = pairs[pair_id]
        rows.append(
            {
                "pair_id": pair_id,
                "condition": response["condition"],
                "prompt_id": pair["prompt_id"],
                "coco_id": pair["coco_id"],
                "question": pair["question"],
                "response": response["response"],
            }
        )

    rng = random.Random(cfg["seed"] + 1009)
    rng.shuffle(rows)

    with sheet_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "annotation_id",
                "prompt_id",
                "coco_id",
                "question",
                "response",
                "score",
                "notes",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "annotation_id": index,
                    "prompt_id": row["prompt_id"],
                    "coco_id": row["coco_id"],
                    "question": row["question"],
                    "response": row["response"],
                    "score": "",
                    "notes": "",
                }
            )

    with key_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["annotation_id", "pair_id", "condition"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "annotation_id": index,
                    "pair_id": row["pair_id"],
                    "condition": row["condition"],
                }
            )

    print(f"Wrote blinded scoring sheet: {sheet_path}")
    print(f"Wrote private answer key:    {key_path}")
    print("Fill score with integers 1-4, then run scripts/merge_human_scores.py.")


if __name__ == "__main__":
    main()
