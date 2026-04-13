"""
Step 2: Pilot Study
- Run PGD on 3 prompts x 1 image = 3 pilot pairs at both epsilon budgets
- Verify PGD loss stabilizes
- Compare clean vs perturbed responses for surrogate-target correlation check
- Save results to results/pilot/
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from _common import (
    apply_style,
    freeze_model,
    generate_response,
    get_target_ids,
    load_config,
    load_llava_model_and_processor,
    pgd_attack,
    pil_to_tensor,
    prepare_prompt_inputs,
    savefig,
    set_seed,
)


def setup_logging(cfg):
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=Path(cfg["output_dir"]) / "errors.log",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def select_pilot_pairs(cfg):
    processed_dir = Path(cfg["data_dir"]) / "processed"
    with (processed_dir / "eval_pairs.json").open() as f:
        eval_pairs = json.load(f)
    with (processed_dir / "prompts.json").open() as f:
        prompts = json.load(f)
    with (processed_dir / "images.json").open() as f:
        images = json.load(f)

    prompt_ids = [prompt["prompt_id"] for prompt in prompts[: cfg["pilot_prompts"]]]
    image_ids = [image["coco_id"] for image in images[: cfg["pilot_images"]]]

    selected_pairs = [
        pair
        for pair in eval_pairs
        if pair["prompt_id"] in prompt_ids and pair["coco_id"] in image_ids
    ]
    selected_pairs.sort(key=lambda pair: (pair["coco_id"], int(pair["prompt_id"])))
    return selected_pairs, image_ids


def fig_pilot_loss_curves(loss_curves, cfg):
    if not loss_curves:
        print("Skipping pilot_loss_curves.png because no successful PGD trials were recorded.")
        return

    eps_labels = sorted({eps for _, eps in loss_curves.keys()})
    fig, axes = plt.subplots(1, len(eps_labels), figsize=(6 * len(eps_labels), 5), squeeze=False)

    for column, eps_label in enumerate(eps_labels):
        ax = axes[0, column]
        for (pair_id, current_eps), curve in sorted(loss_curves.items()):
            if current_eps != eps_label:
                continue
            ax.plot(
                range(1, len(curve) + 1),
                curve,
                label=f"pair {pair_id}",
                linewidth=1.5,
            )
        ax.set_xlabel("PGD step", fontsize=16)
        ax.set_ylabel("Surrogate cross-entropy", fontsize=16)
        ax.legend(fontsize=14)
        ax.text(
            0.95,
            0.95,
            f"$\\epsilon$={eps_label}",
            transform=ax.transAxes,
            fontsize=14,
            ha="right",
            va="top",
        )
        apply_style(ax)

    fig.tight_layout()
    savefig(fig, Path(cfg["output_dir"]) / "figures" / "pilot_loss_curves.png")
    plt.close(fig)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    setup_logging(cfg)

    print("=" * 60)
    print("Step 2: Pilot Study")
    print("=" * 60)

    pilot_pairs, image_ids = select_pilot_pairs(cfg)
    if not pilot_pairs:
        raise RuntimeError("No pilot pairs found. Run preprocess first.")

    print(
        f"Pilot pairs: {len(pilot_pairs)} "
        f"({cfg['pilot_prompts']} prompts x {cfg['pilot_images']} image(s): {', '.join(image_ids)})"
    )

    print(f"\nLoading model: {cfg['model']} ...")
    processor, model = load_llava_model_and_processor(cfg["model"])
    freeze_model(model)
    target_ids = get_target_ids(processor, cfg["surrogate_target"], model.device)

    pilot_dir = Path(cfg["output_dir"]) / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    results = []
    loss_curves = {}
    failed_trials = []
    for pair in pilot_pairs:
        pair_id = pair["pair_id"]
        question = pair["question"]
        image = Image.open(pair["image_path"]).convert("RGB")
        raw_image = pil_to_tensor(image).to(model.device)
        prompt_inputs = prepare_prompt_inputs(processor, question, image, model.device)

        print(f"\n--- Pair {pair_id}: {question[:80]}...")

        pair_result = {
            "pair_id": pair_id,
            "prompt_id": pair["prompt_id"],
            "coco_id": pair["coco_id"],
            "question": question,
            "manual_review_required": True,
            "clean": None,
            "pgd": {},
        }

        try:
            clean_response = generate_response(
                model,
                processor,
                raw_image,
                prompt_inputs["input_ids"],
                prompt_inputs["attention_mask"],
                cfg,
            )
            print(f"  Clean: {clean_response[:120]}...")
            pair_result["clean"] = {
                "response": clean_response,
            }
        except Exception as exc:
            logging.error("pilot clean pair=%s: %s", pair_id, exc, exc_info=True)
            failed_trials.append({"pair_id": pair_id, "condition": "clean", "error": str(exc)})
            print(f"  Clean | ERROR: {exc}")
            results.append(pair_result)
            torch.cuda.empty_cache()
            continue

        for epsilon in cfg["epsilon"]:
            condition_name = f"pgd_{int(round(epsilon * 255))}"
            epsilon_label = f"{epsilon:.4f}"
            alpha = epsilon / 10.0

            try:
                adv_image, curve = pgd_attack(
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
                adv_response = generate_response(
                    model,
                    processor,
                    adv_image,
                    prompt_inputs["input_ids"],
                    prompt_inputs["attention_mask"],
                    cfg,
                )
                loss_curves[(pair_id, epsilon_label)] = curve
                pair_result["pgd"][condition_name] = {
                    "epsilon": epsilon,
                    "loss_curve": curve,
                    "loss_start": round(curve[0], 4),
                    "loss_end": round(curve[-1], 4),
                    "response": adv_response,
                    "comparison": {
                        "clean_response": pair_result["clean"]["response"],
                        "adv_response": adv_response,
                        "manual_assessment": "",
                    },
                }
                print(
                    f"  PGD eps={epsilon_label}: {adv_response[:120]}..."
                )
                print(f"    Loss: start={curve[0]:.4f} end={curve[-1]:.4f}")
            except Exception as exc:
                logging.error(
                    "pilot pgd pair=%s eps=%s: %s",
                    pair_id,
                    epsilon,
                    exc,
                    exc_info=True,
                )
                failed_trials.append(
                    {
                        "pair_id": pair_id,
                        "condition": condition_name,
                        "epsilon": epsilon,
                        "error": str(exc),
                    }
                )
                print(f"  PGD eps={epsilon_label} | ERROR: {exc}")

            torch.cuda.empty_cache()

        results.append(pair_result)

    summary = {
        "model": cfg["model"],
        "surrogate_target": cfg["surrogate_target"],
        "failed_trials": failed_trials,
        "pairs": results,
    }
    results_path = pilot_dir / "pilot_results.json"
    with results_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved pilot results to {results_path}")

    fig_pilot_loss_curves(loss_curves, cfg)

    print("\n" + "=" * 60)
    print("Pilot complete.")
    print(f"  Pairs tested:   {len(results)}")
    if failed_trials:
        print(f"  Failed trials:  {len(failed_trials)} (see results/errors.log)")
    else:
        print("  Failed trials:  0")
    print(f"  Review {results_path} for clean-vs-PGD response comparisons.")
    print("  All main experiment choices remain fixed after this pilot.")
    print("=" * 60)


if __name__ == "__main__":
    main()
