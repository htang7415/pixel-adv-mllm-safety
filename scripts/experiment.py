"""
Step 3: Experiment
- Run all 7 conditions x 45 pairs = 315 trials
- Conditions: clean, random noise (eps 4/255, 8/255), FGSM (eps 4/255, 8/255),
              PGD (eps 4/255, 8/255)
- Compute LPIPS per trial, optionally SSIM/PSNR
- Checkpoint after each condition block; resume on restart
- Produce figures: pgd_loss_curves.png, perturbation_examples.png
"""

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from _common import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    CONDITIONS,
    PALETTE,
    apply_style,
    fgsm_attack,
    freeze_model,
    generate_response,
    get_target_ids,
    load_config,
    load_llava_model_and_processor,
    pgd_attack,
    pil_to_tensor,
    prepare_prompt_inputs,
    random_noise,
    savefig,
    set_seed,
    tensor_to_numpy_uint8,
    tensor_to_pil,
)

try:
    import lpips
except ImportError as exc:
    raise RuntimeError("lpips is required for this project. Install it before running experiment.py.") from exc

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def setup_logging(cfg):
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=Path(cfg["output_dir"]) / "errors.log",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def compute_lpips(lpips_model, clean_image, perturbed_image):
    device = next(lpips_model.parameters()).device
    clean = clean_image.to(device=device, dtype=torch.float32) * 2.0 - 1.0
    perturbed = perturbed_image.to(device=device, dtype=torch.float32) * 2.0 - 1.0
    with torch.no_grad():
        return float(lpips_model(clean, perturbed).item())


def compute_ssim_psnr(clean_image, perturbed_image):
    clean_np = tensor_to_numpy_uint8(clean_image)
    perturbed_np = tensor_to_numpy_uint8(perturbed_image)
    ssim_value = ssim_fn(clean_np, perturbed_np, channel_axis=2, data_range=255)
    psnr_value = psnr_fn(clean_np, perturbed_np, data_range=255)
    return float(ssim_value), float(psnr_value)


def load_pairs(cfg):
    with (Path(cfg["data_dir"]) / "processed" / "eval_pairs.json").open() as f:
        return json.load(f)


def load_checkpoint(cfg):
    checkpoint_path = Path(cfg["output_dir"]) / "checkpoint.json"
    if not checkpoint_path.exists():
        return {
            "completed_conditions": [],
            "responses": [],
            "distortion_rows": [],
            "pgd_loss_rows": [],
            "failed_trials": [],
        }

    with checkpoint_path.open() as f:
        checkpoint = json.load(f)
    checkpoint.setdefault("completed_conditions", [])
    checkpoint.setdefault("responses", [])
    checkpoint.setdefault("distortion_rows", [])
    checkpoint.setdefault("pgd_loss_rows", [])
    checkpoint.setdefault("failed_trials", [])
    return checkpoint


def save_checkpoint(checkpoint, cfg):
    checkpoint_path = Path(cfg["output_dir"]) / "checkpoint.json"
    with checkpoint_path.open("w") as f:
        json.dump(checkpoint, f, indent=2)


def count_rows(rows, condition):
    return sum(1 for row in rows if row["condition"] == condition)


def validate_completed_conditions(checkpoint, pairs):
    expected_trials = len(pairs)
    valid_completed = []
    for condition in checkpoint["completed_conditions"]:
        response_count = count_rows(checkpoint["responses"], condition)
        distortion_count = count_rows(checkpoint["distortion_rows"], condition)
        pgd_ok = True
        if condition.startswith("pgd"):
            pgd_count = count_rows(checkpoint["pgd_loss_rows"], condition)
            pgd_ok = pgd_count == expected_trials

        if response_count == expected_trials and distortion_count == expected_trials and pgd_ok:
            valid_completed.append(condition)
        else:
            print(
                f"Checkpoint inconsistency for {condition}: "
                f"responses={response_count}, distortion={distortion_count}. Re-running it."
            )
    checkpoint["completed_conditions"] = valid_completed


def filter_condition_rows(rows, condition):
    return [row for row in rows if row["condition"] != condition]


def sort_response_rows(rows):
    order = {condition: idx for idx, condition in enumerate(CONDITION_ORDER)}
    return sorted(rows, key=lambda row: (row["pair_id"], order[row["condition"]]))


def sort_distortion_rows(rows):
    order = {condition: idx for idx, condition in enumerate(CONDITION_ORDER)}
    return sorted(rows, key=lambda row: (row["pair_id"], order[row["condition"]]))


def sort_loss_rows(rows):
    order = {condition: idx for idx, condition in enumerate(CONDITION_ORDER)}
    return sorted(rows, key=lambda row: (order[row["condition"]], row["pair_id"]))


def write_outputs(responses, distortion_rows, pgd_loss_rows, cfg):
    output_dir = Path(cfg["output_dir"])

    responses_path = output_dir / "outputs" / "responses.json"
    with responses_path.open("w") as f:
        json.dump(sort_response_rows(responses), f, indent=2)

    distortion_path = output_dir / "metrics" / "distortion.csv"
    with distortion_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "condition",
                "epsilon",
                "lpips",
                "ssim",
                "psnr",
            ],
        )
        writer.writeheader()
        for row in sort_distortion_rows(distortion_rows):
            writer.writerow(row)

    pgd_loss_path = output_dir / "metrics" / "pgd_loss_curves.json"
    with pgd_loss_path.open("w") as f:
        json.dump(sort_loss_rows(pgd_loss_rows), f, indent=2)

    return responses_path, distortion_path, pgd_loss_path


def ensure_human_scores_template(pairs, cfg):
    scores_path = Path(cfg["output_dir"]) / "metrics" / "human_scores.csv"
    if scores_path.exists():
        return scores_path

    with scores_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "condition", "score"])
        for pair in pairs:
            for condition in CONDITION_ORDER:
                writer.writerow([pair["pair_id"], condition, ""])
    return scores_path


def fig_pgd_loss_curves(pgd_loss_rows, cfg):
    if not pgd_loss_rows:
        return

    representative = []
    seen_pairs = set()
    for row in sorted(pgd_loss_rows, key=lambda item: item["pair_id"]):
        if row["pair_id"] in seen_pairs:
            continue
        representative.append(row["pair_id"])
        seen_pairs.add(row["pair_id"])
        if len(representative) == 3:
            break

    fig, ax = plt.subplots(figsize=(8, 5))
    for epsilon_index, condition in enumerate(["pgd_4", "pgd_8"]):
        for row in pgd_loss_rows:
            if row["condition"] != condition or row["pair_id"] not in representative:
                continue
            ax.plot(
                range(1, len(row["loss_curve"]) + 1),
                row["loss_curve"],
                color=PALETTE[epsilon_index + 5],
                alpha=0.75,
                linewidth=1.5,
                label=f"{CONDITION_LABELS[condition]}, pair {row['pair_id']}",
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=12)
    ax.set_xlabel("PGD step", fontsize=16)
    ax.set_ylabel("Surrogate cross-entropy", fontsize=16)
    apply_style(ax)
    savefig(fig, Path(cfg["output_dir"]) / "figures" / "pgd_loss_curves.png")
    plt.close(fig)


def fig_perturbation_examples(pairs, distortion_rows, cfg):
    if not pairs:
        return

    example_pair_id = pairs[0]["pair_id"]
    image_dir = Path(cfg["output_dir"]) / "images"
    distortion_map = {
        (row["pair_id"], row["condition"]): row
        for row in distortion_rows
    }

    available_conditions = []
    for condition in CONDITION_ORDER:
        image_path = image_dir / f"pair{example_pair_id}_{condition}.png"
        if image_path.exists():
            available_conditions.append(condition)

    if len(available_conditions) != len(CONDITION_ORDER):
        print("Skipping perturbation_examples.png because not all condition images are available yet.")
        return

    fig, axes = plt.subplots(1, len(available_conditions), figsize=(4 * len(available_conditions), 4))
    for ax, condition in zip(axes, available_conditions):
        image = Image.open(image_dir / f"pair{example_pair_id}_{condition}.png").convert("RGB")
        ax.imshow(image)
        label = CONDITION_LABELS[condition]
        lpips_value = distortion_map[(example_pair_id, condition)]["lpips"]
        if lpips_value not in (None, ""):
            label += f"\nLPIPS={float(lpips_value):.4f}"
        ax.set_xlabel(label, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    savefig(fig, Path(cfg["output_dir"]) / "figures" / "perturbation_examples.png")
    plt.close(fig)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    setup_logging(cfg)

    print("=" * 60)
    print("Step 3: Experiment")
    print("=" * 60)

    pairs = load_pairs(cfg)
    print(f"Loaded {len(pairs)} eval pairs")

    checkpoint = load_checkpoint(cfg)
    validate_completed_conditions(checkpoint, pairs)
    completed_conditions = set(checkpoint["completed_conditions"])
    print(f"Already completed conditions: {sorted(completed_conditions)}")

    output_dir = Path(cfg["output_dir"])
    for subdir in ["outputs", "images", "metrics", "figures"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {cfg['model']} ...")
    processor, model = load_llava_model_and_processor(cfg["model"])
    freeze_model(model)
    target_ids = get_target_ids(processor, cfg["surrogate_target"], model.device)

    lpips_model = lpips.LPIPS(net="alex").to(model.device)
    lpips_model.eval()

    responses = checkpoint["responses"]
    distortion_rows = checkpoint["distortion_rows"]
    pgd_loss_rows = checkpoint["pgd_loss_rows"]
    failed_trials = checkpoint["failed_trials"]

    for condition, epsilon_index in CONDITIONS:
        if condition in completed_conditions:
            print(f"\nSkipping {condition} (already complete)")
            continue

        responses = filter_condition_rows(responses, condition)
        distortion_rows = filter_condition_rows(distortion_rows, condition)
        pgd_loss_rows = filter_condition_rows(pgd_loss_rows, condition)
        failed_trials = filter_condition_rows(failed_trials, condition)

        epsilon = cfg["epsilon"][epsilon_index] if epsilon_index is not None else None
        alpha = epsilon / 10.0 if epsilon is not None else None
        success_count = 0

        print(f"\n{'-' * 40}")
        print(f"Condition: {condition}")
        print(f"{'-' * 40}")

        for pair_index, pair in enumerate(pairs, start=1):
            pair_id = pair["pair_id"]

            try:
                image = Image.open(pair["image_path"]).convert("RGB")
                raw_image = pil_to_tensor(image).to(model.device)
                prompt_inputs = prepare_prompt_inputs(processor, pair["question"], image, model.device)

                loss_curve = None
                if condition == "clean":
                    perturbed_image = raw_image.clone()
                elif condition.startswith("random"):
                    perturbed_image = random_noise(raw_image, epsilon)
                elif condition.startswith("fgsm"):
                    perturbed_image, loss_curve = fgsm_attack(
                        model,
                        processor,
                        raw_image,
                        prompt_inputs["input_ids"],
                        prompt_inputs["attention_mask"],
                        target_ids,
                        epsilon,
                    )
                elif condition.startswith("pgd"):
                    perturbed_image, loss_curve = pgd_attack(
                        model,
                        processor,
                        raw_image,
                        prompt_inputs["input_ids"],
                        prompt_inputs["attention_mask"],
                        target_ids,
                        epsilon,
                        cfg["pgd_steps"],
                        alpha,
                    )
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                response = generate_response(
                    model,
                    processor,
                    perturbed_image,
                    prompt_inputs["input_ids"],
                    prompt_inputs["attention_mask"],
                    cfg,
                )

                lpips_value = None
                ssim_value = None
                psnr_value = None
                if condition != "clean":
                    lpips_value = compute_lpips(lpips_model, raw_image, perturbed_image)
                    if HAS_SKIMAGE:
                        ssim_value, psnr_value = compute_ssim_psnr(raw_image, perturbed_image)

                image_path = output_dir / "images" / f"pair{pair_id}_{condition}.png"
                tensor_to_pil(perturbed_image).save(image_path)

                responses.append(
                    {
                        "pair_id": pair_id,
                        "prompt_id": pair["prompt_id"],
                        "coco_id": pair["coco_id"],
                        "condition": condition,
                        "response": response,
                    }
                )
                distortion_rows.append(
                    {
                        "pair_id": pair_id,
                        "condition": condition,
                        "epsilon": "" if epsilon is None else epsilon,
                        "lpips": "" if lpips_value is None else f"{lpips_value:.6f}",
                        "ssim": "" if ssim_value is None else f"{ssim_value:.6f}",
                        "psnr": "" if psnr_value is None else f"{psnr_value:.6f}",
                    }
                )
                if condition.startswith("pgd"):
                    pgd_loss_rows.append(
                        {
                            "pair_id": pair_id,
                            "condition": condition,
                            "epsilon": epsilon,
                            "loss_curve": loss_curve,
                        }
                    )

                success_count += 1
                lpips_str = "N/A" if lpips_value is None else f"{lpips_value:.4f}"
                print(
                    f"  [{pair_index}/{len(pairs)}] pair {pair_id} | "
                    f"{condition} | LPIPS={lpips_str}"
                )
            except Exception as exc:
                logging.error("pair=%s condition=%s: %s", pair_id, condition, exc, exc_info=True)
                failed_trials.append(
                    {
                        "pair_id": pair_id,
                        "condition": condition,
                        "error": str(exc),
                    }
                )
                print(f"  [{pair_index}/{len(pairs)}] pair {pair_id} | ERROR: {exc}")

            torch.cuda.empty_cache()

        if success_count == len(pairs):
            if condition not in completed_conditions:
                checkpoint["completed_conditions"].append(condition)
                completed_conditions.add(condition)
            print(f"Completed {condition}: {success_count}/{len(pairs)} trials")
        else:
            print(
                f"Incomplete {condition}: {success_count}/{len(pairs)} successful. "
                "The condition will be retried on the next run."
            )

        checkpoint["responses"] = sort_response_rows(responses)
        checkpoint["distortion_rows"] = sort_distortion_rows(distortion_rows)
        checkpoint["pgd_loss_rows"] = sort_loss_rows(pgd_loss_rows)
        checkpoint["failed_trials"] = failed_trials
        save_checkpoint(checkpoint, cfg)

    responses_path, distortion_path, pgd_loss_path = write_outputs(
        responses,
        distortion_rows,
        pgd_loss_rows,
        cfg,
    )

    incomplete_conditions = [condition for condition in CONDITION_ORDER if condition not in completed_conditions]
    if not incomplete_conditions:
        fig_pgd_loss_curves(pgd_loss_rows, cfg)
        fig_perturbation_examples(pairs, distortion_rows, cfg)
        scores_path = ensure_human_scores_template(pairs, cfg)
    else:
        scores_path = Path(cfg["output_dir"]) / "metrics" / "human_scores.csv"

    print("\n" + "=" * 60)
    print("Experiment status")
    print(f"  Responses saved:   {responses_path}")
    print(f"  Distortion saved:  {distortion_path}")
    print(f"  PGD curves saved:  {pgd_loss_path}")
    if not incomplete_conditions:
        print(f"  Human-score file:  {scores_path}")
        print("  Next step: fill that CSV with 1-4 human scores for all 315 rows, then run bash run_postprocess.sh.")
        print("  Experiment complete.")
    else:
        print(f"  Incomplete conds:  {', '.join(incomplete_conditions)}")
        print("  Experiment is not complete. Re-run to retry failed condition blocks.")
    print("=" * 60)

    if incomplete_conditions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
