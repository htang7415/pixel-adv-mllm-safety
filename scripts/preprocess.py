"""
Step 1: Preprocess
- Sample 15 prompts from one high-severity MM-SafetyBench category
- Select 3 COCO val2017 seed images (indoor, outdoor, object-centered)
- Generate 45 prompt-image pairs
- Verify model loads in fp16 with gradient flow
- Produce figures: prompt_length_distribution.png, seed_images_grid.png
"""

import json
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from _common import (
    PALETTE,
    apply_style,
    compute_surrogate_cross_entropy,
    freeze_model,
    get_target_ids,
    load_config,
    load_llava_model_and_processor,
    pil_to_tensor,
    prepare_prompt_inputs,
    savefig,
    set_seed,
)


SEED_IMAGE_IDS = [
    {"coco_id": "000000000139", "scene_type": "indoor"},
    {"coco_id": "000000000285", "scene_type": "outdoor"},
    {"coco_id": "000000000632", "scene_type": "object-centered"},
]


def sample_prompts(cfg):
    processed_dir = Path(cfg["data_dir"]) / "MM-SafetyBench" / "data" / "processed_questions"
    category_path = processed_dir / f"{cfg['prompt_category']}.json"
    with category_path.open() as f:
        data = json.load(f)

    prompt_ids = sorted(data.keys(), key=int)
    if cfg["num_prompts"] > len(prompt_ids):
        raise ValueError(
            f"Requested {cfg['num_prompts']} prompts but only found {len(prompt_ids)} in {category_path}"
        )

    sampled_ids = sorted(random.sample(prompt_ids, cfg["num_prompts"]), key=int)

    prompts = []
    for prompt_id in sampled_ids:
        entry = data[prompt_id]
        prompts.append(
            {
                "prompt_id": prompt_id,
                "category": cfg["prompt_category"],
                "question": entry["Changed Question"],
            }
        )

    out_dir = Path(cfg["data_dir"]) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prompts.json"
    with out_path.open("w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {out_path}")
    return prompts


def select_images(cfg):
    if cfg["num_images"] > len(SEED_IMAGE_IDS):
        raise ValueError(
            f"Requested {cfg['num_images']} seed images but only {len(SEED_IMAGE_IDS)} are curated."
        )

    coco_dir = Path(cfg["data_dir"]) / "coco" / "val2017"
    out_dir = Path(cfg["data_dir"]) / "processed" / "seed_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = SEED_IMAGE_IDS[: cfg["num_images"]]
    images = []
    for entry in selected:
        src = coco_dir / f"{entry['coco_id']}.jpg"
        if not src.exists():
            raise FileNotFoundError(f"COCO image not found: {src}")
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        images.append(
            {
                "coco_id": entry["coco_id"],
                "scene_type": entry["scene_type"],
                "path": str(dst),
            }
        )

    meta_path = Path(cfg["data_dir"]) / "processed" / "images.json"
    with meta_path.open("w") as f:
        json.dump(images, f, indent=2)
    print(f"Saved {len(images)} image metadata to {meta_path}")
    return images


def generate_pairs(prompts, images, cfg):
    pairs = []
    pair_id = 0
    for prompt in prompts:
        for image in images:
            pairs.append(
                {
                    "pair_id": pair_id,
                    "prompt_id": prompt["prompt_id"],
                    "question": prompt["question"],
                    "coco_id": image["coco_id"],
                    "scene_type": image["scene_type"],
                    "image_path": image["path"],
                }
            )
            pair_id += 1

    out_path = Path(cfg["data_dir"]) / "processed" / "eval_pairs.json"
    with out_path.open("w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} eval pairs to {out_path}")
    return pairs


def verify_model(cfg, processor, model, image_meta):
    image = Image.open(image_meta["path"]).convert("RGB")
    question = "Describe this image."
    prompt_inputs = prepare_prompt_inputs(processor, question, image, model.device)
    target_ids = get_target_ids(processor, cfg["surrogate_target"], model.device)
    raw_image = pil_to_tensor(image).to(model.device).requires_grad_(True)

    model.zero_grad(set_to_none=True)
    loss = compute_surrogate_cross_entropy(
        model,
        processor,
        raw_image,
        prompt_inputs["input_ids"],
        prompt_inputs["attention_mask"],
        target_ids,
    )
    loss.backward()

    grad_norm = raw_image.grad.norm().item()
    if grad_norm <= 0:
        raise AssertionError("Gradients do not flow through the raw image input.")

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm on raw image: {grad_norm:.6f}")
    print("  Gradient flow verified through raw RGB image input.")
def fig_prompt_length_distribution(prompts, processor, cfg):
    lengths = [
        len(processor.tokenizer(prompt["question"], add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        lengths,
        bins=range(min(lengths), max(lengths) + 2),
        color=PALETTE[0],
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel("Prompt length (tokens)", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    apply_style(ax)
    savefig(fig, Path(cfg["output_dir"]) / "figures" / "prompt_length_distribution.png")
    plt.close(fig)


def fig_seed_images_grid(images, cfg):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]

    for ax, image_meta in zip(axes, images):
        image = Image.open(image_meta["path"]).convert("RGB")
        ax.imshow(image)
        ax.set_xlabel(image_meta["scene_type"], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    savefig(fig, Path(cfg["output_dir"]) / "figures" / "seed_images_grid.png")
    plt.close(fig)


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

    print(f"\n[4/5] Loading processor/model: {cfg['model']} ...")
    processor, model = load_llava_model_and_processor(cfg["model"])
    freeze_model(model)

    print("\nVerifying raw-image gradient flow ...")
    verify_model(cfg, processor, model, images[0])

    print("\n[5/5] Generating figures ...")
    fig_prompt_length_distribution(prompts, processor, cfg)
    fig_seed_images_grid(images, cfg)

    print("\n" + "=" * 60)
    print("Preprocess complete.")
    print(f"  Prompts:    {len(prompts)}")
    print(f"  Images:     {len(images)}")
    print(f"  Pairs:      {len(pairs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
