import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
]

CONDITION_ORDER = [
    "clean",
    "random_4",
    "random_8",
    "fgsm_4",
    "fgsm_8",
    "pgd_4",
    "pgd_8",
]

CONDITION_LABELS = {
    "clean": "Clean",
    "random_4": "Random 4/255",
    "random_8": "Random 8/255",
    "fgsm_4": "FGSM 4/255",
    "fgsm_8": "FGSM 8/255",
    "pgd_4": "PGD 4/255",
    "pgd_8": "PGD 8/255",
}

CONDITIONS = [
    ("clean", None),
    ("random_4", 0),
    ("random_8", 1),
    ("fgsm_4", 0),
    ("fgsm_8", 1),
    ("pgd_4", 0),
    ("pgd_8", 1),
]

CONDITION_COLORS = {condition: PALETTE[idx] for idx, condition in enumerate(CONDITION_ORDER)}


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")


def build_prompt(question):
    return f"USER: <image>\n{question}\nASSISTANT:"


def generation_kwargs(cfg):
    decoding = cfg.get("decoding", "greedy")
    temperature = cfg.get("temperature", 0)
    if decoding != "greedy":
        raise ValueError(f"Unsupported decoding='{decoding}'. Project requires greedy decoding.")
    if temperature not in (0, 0.0, None):
        raise ValueError(
            f"temperature={temperature} is inconsistent with greedy decoding for this project."
        )
    return {
        "max_new_tokens": cfg["max_new_tokens"],
        "do_sample": False,
    }


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)


def get_model_dtype(model):
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16


def runtime_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runtime_model_dtype(device):
    return torch.float16 if device.type == "cuda" else torch.float32


def load_llava_model_and_processor(model_name):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    device = runtime_device()
    dtype = runtime_model_dtype(device)

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return processor, model


def reset_peak_memory(device):
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except TypeError:
            torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device):
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.max_memory_allocated(device) / 1e9
    except TypeError:
        return torch.cuda.max_memory_allocated() / 1e9


def pil_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_numpy_uint8(tensor):
    array = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def tensor_to_pil(tensor):
    return Image.fromarray(tensor_to_numpy_uint8(tensor))


def prepare_prompt_inputs(processor, question, image, device):
    prompt_text = build_prompt(question)
    encoded = processor(text=prompt_text, images=image, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def get_target_ids(processor, surrogate_target, device):
    tokens = processor.tokenizer(
        surrogate_target,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return tokens["input_ids"].to(device)


def _normalize_spatial_config(config_value):
    if config_value is None:
        return {}
    if isinstance(config_value, int):
        return {"height": int(config_value), "width": int(config_value)}
    if isinstance(config_value, (tuple, list)) and len(config_value) == 2:
        return {"height": int(config_value[0]), "width": int(config_value[1])}
    if isinstance(config_value, dict):
        return {key: int(value) for key, value in config_value.items()}
    # Handle transformers SizeDict (not a dict subclass)
    result = {}
    for attr in ("height", "width", "shortest_edge", "longest_edge"):
        val = getattr(config_value, attr, None)
        if val is not None:
            result[attr] = int(val)
    if result:
        return result
    return {}


def _resolve_resize_shape(image_processor, height, width):
    size_cfg = _normalize_spatial_config(getattr(image_processor, "size", None))
    if "height" in size_cfg and "width" in size_cfg:
        return size_cfg["height"], size_cfg["width"]
    if "shortest_edge" in size_cfg:
        shortest_edge = size_cfg["shortest_edge"]
        scale = shortest_edge / float(min(height, width))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    if "longest_edge" in size_cfg:
        longest_edge = size_cfg["longest_edge"]
        scale = longest_edge / float(max(height, width))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    return height, width


def _resolve_crop_shape(image_processor):
    crop_cfg = _normalize_spatial_config(getattr(image_processor, "crop_size", None))
    if "height" in crop_cfg and "width" in crop_cfg:
        return crop_cfg["height"], crop_cfg["width"]
    if "shortest_edge" in crop_cfg:
        return crop_cfg["shortest_edge"], crop_cfg["shortest_edge"]
    return None


def _center_crop_with_padding(tensor, crop_height, crop_width):
    _, _, height, width = tensor.shape
    if height < crop_height or width < crop_width:
        pad_height = max(crop_height - height, 0)
        pad_width = max(crop_width - width, 0)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))
        _, _, height, width = tensor.shape

    top = max((height - crop_height) // 2, 0)
    left = max((width - crop_width) // 2, 0)
    return tensor[:, :, top : top + crop_height, left : left + crop_width]


def preprocess_raw_image(raw_image, image_processor, device, model_dtype):
    image = raw_image.to(device=device, dtype=torch.float32)
    _, _, height, width = image.shape

    if getattr(image_processor, "do_resize", True):
        resize_height, resize_width = _resolve_resize_shape(image_processor, height, width)
        if (resize_height, resize_width) != (height, width):
            try:
                image = F.interpolate(
                    image,
                    size=(resize_height, resize_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            except TypeError:
                image = F.interpolate(
                    image,
                    size=(resize_height, resize_width),
                    mode="bilinear",
                    align_corners=False,
                )

    if getattr(image_processor, "do_center_crop", False):
        crop_shape = _resolve_crop_shape(image_processor)
        if crop_shape is not None:
            image = _center_crop_with_padding(image, crop_shape[0], crop_shape[1])

    if getattr(image_processor, "do_normalize", True):
        mean = torch.tensor(
            image_processor.image_mean,
            device=device,
            dtype=image.dtype,
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            image_processor.image_std,
            device=device,
            dtype=image.dtype,
        ).view(1, -1, 1, 1)
        image = (image - mean) / std

    return image.to(dtype=model_dtype)


def compute_surrogate_cross_entropy(
    model,
    processor,
    raw_image,
    input_ids,
    attention_mask,
    target_ids,
):
    batch_size = input_ids.shape[0]
    combined_ids = torch.cat([input_ids, target_ids.expand(batch_size, -1)], dim=1)
    combined_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                batch_size,
                target_ids.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ],
        dim=1,
    )

    labels = combined_ids.clone()
    labels[:, : input_ids.shape[1]] = -100

    pixel_values = preprocess_raw_image(
        raw_image,
        processor.image_processor,
        input_ids.device,
        get_model_dtype(model),
    )
    outputs = model(
        input_ids=combined_ids,
        attention_mask=combined_mask,
        pixel_values=pixel_values,
        labels=labels,
    )
    return outputs.loss


def random_noise(raw_image, epsilon):
    noise = torch.empty_like(raw_image).uniform_(-epsilon, epsilon)
    perturbed = raw_image + noise
    delta = torch.clamp(perturbed - raw_image, -epsilon, epsilon)
    return torch.clamp(raw_image + delta, 0.0, 1.0)


def fgsm_attack(model, processor, raw_image, input_ids, attention_mask, target_ids, epsilon):
    x_orig = raw_image.detach()
    x_adv = x_orig.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    loss = compute_surrogate_cross_entropy(
        model,
        processor,
        x_adv,
        input_ids,
        attention_mask,
        target_ids,
    )
    loss.backward()

    x_adv = x_adv.detach() - epsilon * x_adv.grad.detach().sign()
    delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
    return torch.clamp(x_orig + delta, 0.0, 1.0), [float(loss.item())]


def pgd_attack(
    model,
    processor,
    raw_image,
    input_ids,
    attention_mask,
    target_ids,
    epsilon,
    steps,
    alpha,
):
    x_orig = raw_image.detach()
    x_adv = x_orig.clone().detach()
    loss_curve = []

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        loss = compute_surrogate_cross_entropy(
            model,
            processor,
            x_adv,
            input_ids,
            attention_mask,
            target_ids,
        )
        loss.backward()

        x_adv = x_adv.detach() - alpha * x_adv.grad.detach().sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        loss_curve.append(float(loss.item()))

    return x_adv.detach(), loss_curve


def generate_response(model, processor, raw_image, input_ids, attention_mask, cfg):
    pixel_values = preprocess_raw_image(
        raw_image,
        processor.image_processor,
        input_ids.device,
        get_model_dtype(model),
    )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **generation_kwargs(cfg),
        )
    generated = output_ids[0, input_ids.shape[1] :]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)
