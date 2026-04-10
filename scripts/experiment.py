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

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


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

CONDITION_COLORS = {
    "clean": PALETTE[0],
    "random_4": PALETTE[1], "random_8": PALETTE[2],
    "fgsm_4": PALETTE[3], "fgsm_8": PALETTE[4],
    "pgd_4": PALETTE[5], "pgd_8": PALETTE[6],
}

CONDITIONS = [
    ("clean", None),
    ("random_4", 0), ("random_8", 1),
    ("fgsm_4", 0), ("fgsm_8", 1),
    ("pgd_4", 0), ("pgd_8", 1),
]


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


def setup_logging(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cfg["output_dir"], "errors.log"),
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )


# ── Model / Generation ───────────────────────────────────────────────────────

def build_prompt(question):
    return f"USER: <image>\n{question}\nASSISTANT:"


def get_target_ids(processor, surrogate_target, device):
    tok = processor.tokenizer(surrogate_target, return_tensors="pt",
                              add_special_tokens=False)
    return tok["input_ids"].to(device)


def generate_response(model, processor, pixel_values, input_ids,
                      attention_mask, cfg):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = output_ids[0, input_ids.shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


# ── Distortion metrics ───────────────────────────────────────────────────────

def compute_lpips(lpips_model, img_clean, img_pert):
    """Compute LPIPS between two image tensors [1,C,H,W] in [0,1]."""
    # LPIPS expects [-1, 1]
    a = img_clean * 2 - 1
    b = img_pert * 2 - 1
    with torch.no_grad():
        return lpips_model(a, b).item()


def compute_ssim_psnr(img_clean_np, img_pert_np):
    """Compute SSIM and PSNR from numpy HWC uint8 images."""
    s = ssim_fn(img_clean_np, img_pert_np, channel_axis=2, data_range=255)
    p = psnr_fn(img_clean_np, img_pert_np, data_range=255)
    return s, p


def tensor_to_numpy_uint8(t):
    """Convert [1,C,H,W] float tensor to HWC uint8 numpy."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    return (arr * 255).astype(np.uint8)


# ── Attack methods ────────────────────────────────────────────────────────────

def random_noise(image_tensor, epsilon):
    noise = torch.empty_like(image_tensor).uniform_(-epsilon, epsilon)
    return torch.clamp(image_tensor + noise, 0.0, 1.0)


def fgsm_attack(model, processor, image_tensor, input_ids, attention_mask,
                target_ids, epsilon):
    x = image_tensor.clone().detach().requires_grad_(True)

    combined_ids = torch.cat([input_ids, target_ids], dim=1)
    combined_mask = torch.cat([
        attention_mask,
        torch.ones(1, target_ids.shape[1],
                   dtype=attention_mask.dtype, device=attention_mask.device)
    ], dim=1)
    labels = combined_ids.clone()
    labels[:, :input_ids.shape[1]] = -100

    outputs = model(
        input_ids=combined_ids, attention_mask=combined_mask,
        pixel_values=x, labels=labels,
    )
    loss = -outputs.loss  # negate: we maximize P(target)
    loss.backward()

    x_adv = image_tensor + epsilon * x.grad.data.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def pgd_attack(model, processor, image_tensor, input_ids, attention_mask,
               target_ids, epsilon, steps, alpha):
    x_adv = image_tensor.clone().detach()
    x_orig = image_tensor.clone().detach()
    loss_curve = []

    for step in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        combined_ids = torch.cat([input_ids, target_ids], dim=1)
        combined_mask = torch.cat([
            attention_mask,
            torch.ones(1, target_ids.shape[1],
                       dtype=attention_mask.dtype, device=attention_mask.device)
        ], dim=1)
        labels = combined_ids.clone()
        labels[:, :input_ids.shape[1]] = -100

        outputs = model(
            input_ids=combined_ids, attention_mask=combined_mask,
            pixel_values=x_adv, labels=labels,
        )
        loss = -outputs.loss
        loss.backward()

        x_adv = x_adv.detach() + alpha * x_adv.grad.data.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        loss_curve.append(outputs.loss.item())  # positive CE

    return x_adv.detach(), loss_curve


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(cfg):
    ckpt_path = os.path.join(cfg["output_dir"], "checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {"completed_conditions": [], "responses": [], "distortion_rows": []}


def save_checkpoint(ckpt, cfg):
    ckpt_path = os.path.join(cfg["output_dir"], "checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    setup_logging(cfg)

    print("=" * 60)
    print("Step 3: Experiment")
    print("=" * 60)

    # Load pairs
    with open(os.path.join(cfg["data_dir"], "processed", "eval_pairs.json")) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} eval pairs")

    # Load checkpoint
    ckpt = load_checkpoint(cfg)
    completed = set(ckpt["completed_conditions"])
    print(f"Already completed conditions: {sorted(completed)}")

    # Setup output dirs
    for subdir in ["outputs", "images", "metrics", "figures"]:
        os.makedirs(os.path.join(cfg["output_dir"], subdir), exist_ok=True)

    # Load model
    print(f"\nLoading model: {cfg['model']} ...")
    processor = AutoProcessor.from_pretrained(cfg["model"])
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    device = model.device
    target_ids = get_target_ids(processor, cfg["surrogate_target"], device)

    # LPIPS model
    lpips_model = None
    if HAS_LPIPS:
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()
    else:
        print("WARNING: lpips not installed. LPIPS will not be computed.")

    # Storage for figures
    pgd_loss_data = {}   # (pair_id, eps_label) -> loss_curve
    example_pair_id = pairs[0]["pair_id"]
    example_images = {}  # condition_name -> (pil_image, lpips_val)

    responses = list(ckpt["responses"])
    distortion_rows = list(ckpt["distortion_rows"])

    for cond_name, eps_idx in CONDITIONS:
        if cond_name in completed:
            print(f"\nSkipping {cond_name} (already completed)")
            continue

        eps = cfg["epsilon"][eps_idx] if eps_idx is not None else None
        eps_label = f"{eps:.4f}" if eps is not None else "none"
        alpha = eps / 10.0 if eps is not None else None

        print(f"\n{'─'*40}")
        print(f"Condition: {cond_name}  (eps={eps_label})")
        print(f"{'─'*40}")

        for i, pair in enumerate(pairs):
            pid = pair["pair_id"]
            question = pair["question"]
            t0 = time.time()

            try:
                image = Image.open(pair["image_path"]).convert("RGB")
                prompt_text = build_prompt(question)
                inputs = processor(text=prompt_text, images=image,
                                   return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                clean_pv = inputs["pixel_values"].clone().detach()

                # Apply perturbation
                loss_curve = None
                if cond_name == "clean":
                    pert_pv = clean_pv
                elif cond_name.startswith("random"):
                    pert_pv = random_noise(clean_pv, eps)
                elif cond_name.startswith("fgsm"):
                    pert_pv = fgsm_attack(
                        model, processor, clean_pv,
                        inputs["input_ids"], inputs["attention_mask"],
                        target_ids, eps,
                    )
                elif cond_name.startswith("pgd"):
                    pert_pv, loss_curve = pgd_attack(
                        model, processor, clean_pv,
                        inputs["input_ids"], inputs["attention_mask"],
                        target_ids, eps, cfg["pgd_steps"], alpha,
                    )
                    pgd_loss_data[(pid, eps_label)] = loss_curve

                # Generate response
                resp = generate_response(
                    model, processor, pert_pv,
                    inputs["input_ids"], inputs["attention_mask"], cfg,
                )

                # Compute distortion
                lpips_val = None
                ssim_val = None
                psnr_val = None

                if cond_name != "clean":
                    if lpips_model is not None:
                        lpips_val = compute_lpips(lpips_model, clean_pv, pert_pv)
                    if HAS_SKIMAGE:
                        clean_np = tensor_to_numpy_uint8(clean_pv)
                        pert_np = tensor_to_numpy_uint8(pert_pv)
                        ssim_val, psnr_val = compute_ssim_psnr(clean_np, pert_np)

                # Save perturbed image
                pert_pil = Image.fromarray(tensor_to_numpy_uint8(pert_pv))
                img_path = os.path.join(
                    cfg["output_dir"], "images",
                    f"pair{pid}_{cond_name}.png"
                )
                pert_pil.save(img_path)

                # Track example pair for figure
                if pid == example_pair_id:
                    example_images[cond_name] = (pert_pil, lpips_val)

                elapsed = time.time() - t0
                responses.append({
                    "pair_id": pid,
                    "condition": cond_name,
                    "response": resp,
                })
                distortion_rows.append({
                    "pair_id": pid,
                    "condition": cond_name,
                    "epsilon": eps if eps else 0,
                    "lpips": lpips_val,
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                })

                print(f"  [{i+1}/{len(pairs)}] pair {pid} | "
                      f"{cond_name} | {elapsed:.1f}s | "
                      f"LPIPS={lpips_val if lpips_val else 'N/A':.4f}"
                      if lpips_val else
                      f"  [{i+1}/{len(pairs)}] pair {pid} | "
                      f"{cond_name} | {elapsed:.1f}s")

            except Exception as e:
                logging.error(f"pair={pid} cond={cond_name}: {e}",
                              exc_info=True)
                print(f"  [{i+1}/{len(pairs)}] pair {pid} | ERROR: {e}")

            torch.cuda.empty_cache()

        # Checkpoint after each condition block
        ckpt["completed_conditions"].append(cond_name)
        ckpt["responses"] = responses
        ckpt["distortion_rows"] = distortion_rows
        save_checkpoint(ckpt, cfg)
        completed.add(cond_name)
        print(f"  Checkpoint saved after {cond_name}")

    # ── Save final outputs ────────────────────────────────────────────────

    resp_path = os.path.join(cfg["output_dir"], "outputs", "responses.json")
    with open(resp_path, "w") as f:
        json.dump(responses, f, indent=2)
    print(f"\nSaved {len(responses)} responses to {resp_path}")

    dist_path = os.path.join(cfg["output_dir"], "metrics", "distortion.csv")
    if distortion_rows:
        fieldnames = ["pair_id", "condition", "epsilon", "lpips", "ssim", "psnr"]
        with open(dist_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(distortion_rows)
        print(f"Saved distortion metrics to {dist_path}")

    # ── Figure: PGD loss curves ───────────────────────────────────────────
    if pgd_loss_data:
        # Pick a few representative pairs
        all_pids = sorted(set(pid for pid, _ in pgd_loss_data.keys()))
        rep_pids = all_pids[:3]  # first 3 pairs
        eps_labels = sorted(set(el for _, el in pgd_loss_data.keys()))

        fig, ax = plt.subplots(figsize=(8, 5))
        for el_idx, el in enumerate(eps_labels):
            for pid in rep_pids:
                if (pid, el) in pgd_loss_data:
                    curve = pgd_loss_data[(pid, el)]
                    ax.plot(range(1, len(curve) + 1), curve,
                            color=PALETTE[el_idx * 3],
                            alpha=0.7, linewidth=1.5,
                            label=f"$\\epsilon$={el}, pair {pid}")
        ax.set_xlabel("PGD step", fontsize=16)
        ax.set_ylabel("Surrogate loss (CE)", fontsize=16)
        ax.legend(fontsize=12, loc="best")
        apply_style(ax)
        savefig(fig, os.path.join(cfg["output_dir"], "figures",
                                  "pgd_loss_curves.png"))

    # ── Figure: perturbation examples ─────────────────────────────────────
    if example_images:
        cond_order = ["clean", "random_4", "random_8",
                      "fgsm_4", "fgsm_8", "pgd_4", "pgd_8"]
        available = [c for c in cond_order if c in example_images]
        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, cond in zip(axes, available):
            pil_img, lp = example_images[cond]
            ax.imshow(pil_img)
            label = cond
            if lp is not None:
                label += f"\nLPIPS={lp:.4f}"
            ax.set_xlabel(label, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.tight_layout()
        savefig(fig, os.path.join(cfg["output_dir"], "figures",
                                  "perturbation_examples.png"))

    # ── Generate blank human_scores.csv template ──────────────────────────
    scores_path = os.path.join(cfg["output_dir"], "metrics", "human_scores.csv")
    if not os.path.exists(scores_path):
        with open(scores_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "condition", "score"])
            for row in distortion_rows:
                writer.writerow([row["pair_id"], row["condition"], ""])
            # Also add clean rows
            for pair in pairs:
                if not any(r["pair_id"] == pair["pair_id"] and
                           r["condition"] == "clean"
                           for r in distortion_rows):
                    writer.writerow([pair["pair_id"], "clean", ""])
        print(f"Generated blank human_scores.csv template at {scores_path}")

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print(f"  Total responses: {len(responses)}")
    print(f"  Fill in {scores_path} with human scores (1-4) for postprocessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
