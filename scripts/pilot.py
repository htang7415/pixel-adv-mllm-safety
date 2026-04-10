"""
Step 2: Pilot Study
- Run PGD on 3 prompts x 1 image = 3 pilot pairs at both epsilon budgets
- Verify PGD loss stabilizes
- Log peak GPU memory and per-trial runtime
- Compare clean vs perturbed responses for surrogate-target correlation check
- Save results to results/pilot/
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3"]

def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path}")


# ── Attack utilities ──────────────────────────────────────────────────────────

def build_prompt(question):
    return f"USER: <image>\n{question}\nASSISTANT:"


def get_target_ids(processor, surrogate_target, device):
    """Tokenize the surrogate target prefix."""
    tok = processor.tokenizer(surrogate_target, return_tensors="pt",
                              add_special_tokens=False)
    return tok["input_ids"].to(device)


def compute_surrogate_loss(model, processor, pixel_values, input_ids,
                           attention_mask, target_ids):
    """
    Compute negative log-likelihood of the surrogate target tokens
    given the image + prompt. Returns scalar loss (to maximize).
    """
    # Append target tokens to input
    batch_size = input_ids.shape[0]
    combined_ids = torch.cat([input_ids, target_ids.expand(batch_size, -1)], dim=1)
    combined_mask = torch.cat([
        attention_mask,
        torch.ones(batch_size, target_ids.shape[1],
                   dtype=attention_mask.dtype, device=attention_mask.device)
    ], dim=1)

    # Labels: -100 for prompt tokens, actual ids for target tokens
    labels = combined_ids.clone()
    labels[:, :input_ids.shape[1]] = -100

    outputs = model(
        input_ids=combined_ids,
        attention_mask=combined_mask,
        pixel_values=pixel_values,
        labels=labels,
    )
    # model returns cross-entropy loss (we want to minimize it to maximize
    # the probability of the target, so we negate for PGD which maximizes)
    return -outputs.loss


def pgd_attack(model, processor, image_tensor, input_ids, attention_mask,
               target_ids, epsilon, steps, alpha, device):
    """
    PGD attack in pixel space. Returns perturbed image tensor and loss curve.
    """
    # Clone and require grad
    x_adv = image_tensor.clone().detach()
    x_orig = image_tensor.clone().detach()
    loss_curve = []

    for step in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        loss = compute_surrogate_loss(
            model, processor, x_adv, input_ids, attention_mask, target_ids
        )
        loss.backward()

        # PGD step: maximize loss (sign ascent on the negated CE)
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv.detach() + alpha * grad_sign

        # Project back to epsilon ball around original
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        loss_curve.append(-loss.item())  # store positive CE loss

    return x_adv.detach(), loss_curve


def generate_response(model, processor, image, question, cfg, device):
    """Generate a response using greedy decoding."""
    prompt_text = build_prompt(question)
    inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    # Decode only the generated tokens
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    print("=" * 60)
    print("Step 2: Pilot Study")
    print("=" * 60)

    # Load pilot pairs
    with open(os.path.join(cfg["data_dir"], "processed", "eval_pairs.json")) as f:
        all_pairs = json.load(f)

    with open(os.path.join(cfg["data_dir"], "processed", "images.json")) as f:
        images_meta = json.load(f)

    # Pilot: first image, first N prompts
    pilot_image = images_meta[0]
    pilot_pairs = [p for p in all_pairs
                   if p["coco_id"] == pilot_image["coco_id"]]
    pilot_pairs = pilot_pairs[:cfg["pilot_prompts"]]
    print(f"Pilot pairs: {len(pilot_pairs)} (image: {pilot_image['coco_id']})")

    # Load model
    print(f"\nLoading model: {cfg['model']} ...")
    processor = AutoProcessor.from_pretrained(cfg["model"])
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    device = model.device

    target_ids = get_target_ids(processor, cfg["surrogate_target"], device)

    # Output directory
    pilot_dir = os.path.join(cfg["output_dir"], "pilot")
    os.makedirs(pilot_dir, exist_ok=True)

    results = []
    all_loss_curves = {}  # (pair_id, eps_label) -> curve

    for pair in pilot_pairs:
        pid = pair["pair_id"]
        question = pair["question"]
        image = Image.open(pair["image_path"]).convert("RGB")

        print(f"\n--- Pair {pid}: {question[:60]}...")

        # Clean response
        t0 = time.time()
        clean_resp = generate_response(model, processor, image, question, cfg, device)
        clean_time = time.time() - t0
        print(f"  Clean response ({clean_time:.1f}s): {clean_resp[:100]}...")

        pair_result = {
            "pair_id": pid,
            "question": question,
            "clean_response": clean_resp,
            "clean_time_s": round(clean_time, 2),
            "pgd": {},
        }

        # PGD at each epsilon
        for eps in cfg["epsilon"]:
            eps_label = f"{eps:.4f}"
            alpha = eps / 10.0
            steps = cfg["pgd_steps"]

            # Prepare image tensor
            prompt_text = build_prompt(question)
            inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            image_tensor = inputs["pixel_values"].clone().detach()

            t0 = time.time()
            x_adv, loss_curve = pgd_attack(
                model, processor, image_tensor,
                inputs["input_ids"], inputs["attention_mask"],
                target_ids, eps, steps, alpha, device,
            )
            attack_time = time.time() - t0

            # Generate response with adversarial image
            inputs["pixel_values"] = x_adv
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg["max_new_tokens"],
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            adv_resp = processor.tokenizer.decode(generated, skip_special_tokens=True)

            print(f"  PGD eps={eps_label} ({attack_time:.1f}s): {adv_resp[:100]}...")
            print(f"    Loss: start={loss_curve[0]:.4f} end={loss_curve[-1]:.4f}")

            pair_result["pgd"][eps_label] = {
                "adv_response": adv_resp,
                "attack_time_s": round(attack_time, 2),
                "loss_start": round(loss_curve[0], 4),
                "loss_end": round(loss_curve[-1], 4),
            }
            all_loss_curves[(pid, eps_label)] = loss_curve

            torch.cuda.empty_cache()

        results.append(pair_result)

    # Peak memory
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak GPU memory: {peak_mem:.2f} GB")

    # Save results
    summary = {
        "peak_gpu_memory_gb": round(peak_mem, 2),
        "pairs": results,
    }
    results_path = os.path.join(pilot_dir, "pilot_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved pilot results to {results_path}")

    # ── Figure: pilot loss curves ─────────────────────────────────────────
    eps_labels = sorted(set(e for _, e in all_loss_curves.keys()))
    fig, axes = plt.subplots(1, len(eps_labels), figsize=(6 * len(eps_labels), 5),
                             squeeze=False)

    for col, eps_label in enumerate(eps_labels):
        ax = axes[0, col]
        for (pid, el), curve in all_loss_curves.items():
            if el == eps_label:
                ax.plot(range(1, len(curve) + 1), curve,
                        label=f"pair {pid}", linewidth=1.5)
        ax.set_xlabel("PGD step", fontsize=16)
        ax.set_ylabel("Surrogate loss (CE)", fontsize=16)
        ax.legend(fontsize=14)
        apply_style(ax)
        # Add eps as text annotation
        ax.text(0.95, 0.95, f"$\\epsilon$={eps_label}",
                transform=ax.transAxes, fontsize=14,
                ha="right", va="top")

    savefig(fig, os.path.join(cfg["output_dir"], "figures",
                              "pilot_loss_curves.png"))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pilot complete.")
    print(f"  Pairs tested:  {len(results)}")
    print(f"  Peak GPU mem:  {peak_mem:.2f} GB")
    print("  Review results/pilot/pilot_results.json for response comparison.")
    print("  All main experiment choices are now frozen.")
    print("=" * 60)


if __name__ == "__main__":
    main()
