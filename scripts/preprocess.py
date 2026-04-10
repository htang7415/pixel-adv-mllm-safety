"""
Step 1: Preprocess
- Sample 15 prompts from one high-severity MM-SafetyBench category
- Select 3 COCO val2017 seed images (indoor, outdoor, object-centered)
- Generate 45 prompt-image pairs
- Verify model loads in fp16 with gradient flow
- Produce figures: prompt_length_distribution.png, seed_images_grid.png
"""

import json
import os
import random
import shutil

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Figure style ──────────────────────────────────────────────────────────────

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


# ── 1. Sample prompts ────────────────────────────────────────────────────────

def sample_prompts(cfg):
    cat_file = os.path.join(
        cfg["data_dir"], "MM-SafetyBench", "data", "processed_questions",
        f"{cfg['prompt_category']}.json"
    )
    with open(cat_file) as f:
        data = json.load(f)

    ids = sorted(data.keys(), key=int)
    sampled = random.sample(ids, cfg["num_prompts"])
    sampled.sort(key=int)

    prompts = []
    for pid in sampled:
        entry = data[pid]
        prompts.append({
            "prompt_id": pid,
            "category": cfg["prompt_category"],
            "question": entry["Changed Question"],
        })

    out_dir = os.path.join(cfg["data_dir"], "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prompts.json")
    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {out_path}")
    return prompts


# ── 2. Select seed images ────────────────────────────────────────────────────

# Hand-picked COCO val2017 IDs for three scene types:
#   indoor  : 000000000139 (living room scene)
#   outdoor : 000000000285 (outdoor scene with animals)
#   object  : 000000000632 (object-centered scene)
SEED_IMAGE_IDS = [
    {"coco_id": "000000000139", "scene_type": "indoor"},
    {"coco_id": "000000000285", "scene_type": "outdoor"},
    {"coco_id": "000000000632", "scene_type": "object-centered"},
]


def select_images(cfg):
    coco_dir = os.path.join(cfg["data_dir"], "coco", "val2017")
    out_dir = os.path.join(cfg["data_dir"], "processed", "seed_images")
    os.makedirs(out_dir, exist_ok=True)

    images = []
    for entry in SEED_IMAGE_IDS:
        src = os.path.join(coco_dir, f"{entry['coco_id']}.jpg")
        if not os.path.exists(src):
            raise FileNotFoundError(f"COCO image not found: {src}")
        dst = os.path.join(out_dir, f"{entry['coco_id']}.jpg")
        shutil.copy2(src, dst)
        images.append({
            "coco_id": entry["coco_id"],
            "scene_type": entry["scene_type"],
            "path": dst,
        })

    meta_path = os.path.join(cfg["data_dir"], "processed", "images.json")
    with open(meta_path, "w") as f:
        json.dump(images, f, indent=2)
    print(f"Saved {len(images)} image metadata to {meta_path}")
    return images


# ── 3. Generate evaluation pairs ─────────────────────────────────────────────

def generate_pairs(prompts, images, cfg):
    pairs = []
    pair_id = 0
    for p in prompts:
        for img in images:
            pairs.append({
                "pair_id": pair_id,
                "prompt_id": p["prompt_id"],
                "question": p["question"],
                "coco_id": img["coco_id"],
                "scene_type": img["scene_type"],
                "image_path": img["path"],
            })
            pair_id += 1

    out_path = os.path.join(cfg["data_dir"], "processed", "eval_pairs.json")
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} eval pairs to {out_path}")
    return pairs


# ── 4. Verify model ──────────────────────────────────────────────────────────

def verify_model(cfg, images):
    print(f"\nLoading model: {cfg['model']} ...")
    processor = AutoProcessor.from_pretrained(cfg["model"])
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    # Check gradient flow through image input
    img = Image.open(images[0]["path"]).convert("RGB")
    prompt_text = "USER: <image>\nDescribe this image.\nASSISTANT:"
    inputs = processor(text=prompt_text, images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Enable gradients on pixel values
    pixel_values = inputs["pixel_values"].clone().detach().requires_grad_(True)
    inputs["pixel_values"] = pixel_values

    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()

    grad_norm = pixel_values.grad.norm().item()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm on pixel_values: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradients do not flow through image input!"
    print("  Gradient flow verified.")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    del model, processor
    torch.cuda.empty_cache()
    return peak_mem


# ── 5. Figures ────────────────────────────────────────────────────────────────

def fig_prompt_length_distribution(prompts, cfg):
    """Histogram of prompt token lengths (word-level tokenization as proxy)."""
    lengths = [len(p["question"].split()) for p in prompts]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=range(min(lengths), max(lengths) + 2),
            color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Prompt length (words)", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    apply_style(ax)
    savefig(fig, os.path.join(cfg["output_dir"], "figures",
                              "prompt_length_distribution.png"))


def fig_seed_images_grid(images, cfg):
    """1x3 grid of seed images with scene-type labels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img_meta in zip(axes, images):
        img = Image.open(img_meta["path"]).convert("RGB")
        ax.imshow(img)
        ax.set_xlabel(img_meta["scene_type"], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    savefig(fig, os.path.join(cfg["output_dir"], "figures",
                              "seed_images_grid.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    print("=" * 60)
    print("Step 1: Preprocess")
    print("=" * 60)

    print("\n[1/5] Sampling prompts ...")
    prompts = sample_prompts(cfg)

    print("\n[2/5] Selecting seed images ...")
    images = select_images(cfg)

    print("\n[3/5] Generating evaluation pairs ...")
    pairs = generate_pairs(prompts, images, cfg)

    print("\n[4/5] Verifying model ...")
    peak_mem = verify_model(cfg, images)

    print("\n[5/5] Generating figures ...")
    fig_prompt_length_distribution(prompts, cfg)
    fig_seed_images_grid(images, cfg)

    print("\n" + "=" * 60)
    print("Preprocess complete.")
    print(f"  Prompts:    {len(prompts)}")
    print(f"  Images:     {len(images)}")
    print(f"  Pairs:      {len(pairs)}")
    print(f"  Peak GPU:   {peak_mem:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
