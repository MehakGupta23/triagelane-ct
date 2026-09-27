import os
import numpy as np
import torch
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CONFIG
from .model import load_model
from .preprocessing import (
    apply_window,
    three_window_stack,
    to_hounsfield,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID2LABEL = {
    int(index): label
    for index, label in CONFIG["id2label"].items()
}

BRAIN_WINDOW = (
    CONFIG["windows"]["brain"]["center"],
    CONFIG["windows"]["brain"]["width"],
)

GRADCAM_LOCALIZATION_THRESHOLD = CONFIG["gradcam"]["localization_threshold"]


class GradCAMTargetLayerError(RuntimeError):
    pass


def _dominant_subtype_class_index(dominant_subtype):
    for index, label in ID2LABEL.items():
        if label == dominant_subtype:
            return index
    raise KeyError(dominant_subtype)


def _locate_transformer_backbone(model):
    for attr_name in ("vit", "deit", "beit", "data2vec_vision", "swin", "swinv2"):
        backbone = getattr(model, attr_name, None)
        if backbone is not None:
            return attr_name, backbone
    return None, None


def _last_transformer_block(container):
    for attr_name in ("layer", "layers"):
        module_list = getattr(container, attr_name, None)
        if module_list is not None and len(module_list) > 0:
            candidate = module_list[-1]
            if hasattr(candidate, "blocks") and len(candidate.blocks) > 0:
                return candidate.blocks[-1]
            return candidate
    return None


def find_gradcam_target_layer(model):
    backbone_name, backbone = _locate_transformer_backbone(model)

    if backbone is not None:
        for container in (
            getattr(backbone, "encoder", None),
            backbone,
        ):
            if container is None:
                continue

            last_block = _last_transformer_block(container)

            if last_block is not None:
                if hasattr(last_block, "layernorm_before"):
                    return (
                        last_block.layernorm_before,
                        f"{backbone_name} backbone, last transformer block, "
                        f"layernorm_before",
                    )

                return (
                    last_block,
                    f"{backbone_name} backbone, last transformer block",
                )

    conv_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]

    if conv_layers:
        name, module = conv_layers[-1]
        return module, name

    raise GradCAMTargetLayerError(
        f"could not automatically identify a Grad-CAM target layer for model type "
        f"{type(model).__name__}; inspect model.named_modules() and select one explicitly"
    )


def _reshape_transformer_activation(tensor):
    if tensor.dim() != 3:
        return tensor

    num_tokens = tensor.shape[1]
    side = int(round(num_tokens ** 0.5))

    if side * side == num_tokens:
        grid_tokens = tensor
    else:
        side = int(round((num_tokens - 1) ** 0.5))

        if side * side == num_tokens - 1:
            grid_tokens = tensor[:, 1:, :]
        else:
            return tensor

    grid = grid_tokens.reshape(
        tensor.shape[0],
        side,
        side,
        tensor.shape[2],
    )

    return grid.permute(0, 3, 1, 2)


def compute_gradcam(
    pixel_array,
    rescale_slope,
    rescale_intercept,
    target_class_index,
):
    extractor, model = load_model()

    stacked = three_window_stack(
        pixel_array,
        rescale_slope,
        rescale_intercept,
    )

    inputs = extractor(
        images=Image.fromarray(stacked),
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)
    pixel_values.requires_grad_(True)

    target_layer, target_layer_name = find_gradcam_target_layer(model)

    activations = {}
    gradients = {}

    def forward_hook(module, layer_inputs, layer_output):
        activations["value"] = layer_output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    model.zero_grad()

    try:
        logits = model(pixel_values=pixel_values).logits[0]
        logits[target_class_index].backward()

        raw_activations = activations["value"].detach()
        raw_gradients = gradients["value"]

        spatial_activations = _reshape_transformer_activation(
            raw_activations
        )
        spatial_gradients = _reshape_transformer_activation(
            raw_gradients
        )

    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad()

    if (
        spatial_activations.dim() != 4
        or spatial_gradients.dim() != 4
    ):
        raise GradCAMTargetLayerError(
            f"target layer '{target_layer_name}' output could not be reshaped "
            f"into a spatial (batch, channels, height, width) map -- "
            f"raw shape was {tuple(raw_activations.shape)}"
        )

    weights = spatial_gradients.mean(
        dim=(2, 3),
        keepdim=True,
    )

    cam = torch.relu(
        (weights * spatial_activations).sum(
            dim=1,
            keepdim=True,
        )
    )

    cam = (
        cam[0, 0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    cam_min = float(cam.min())
    cam_max = float(cam.max())

    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    native_height, native_width = pixel_array.shape

    cam_resized = np.array(
        Image.fromarray(
            (cam * 255).astype(np.uint8)
        ).resize(
            (native_width, native_height),
            resample=Image.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0

    mask = cam_resized >= GRADCAM_LOCALIZATION_THRESHOLD

    if mask.any():
        rows, cols = np.where(mask)

        weights_at_mask = cam_resized[
            rows,
            cols,
        ]

        bounding_box = {
            "row_min": int(rows.min()),
            "row_max": int(rows.max()),
            "col_min": int(cols.min()),
            "col_max": int(cols.max()),
        }

        centroid = {
            "row": float(
                (rows * weights_at_mask).sum()
                / weights_at_mask.sum()
            ),
            "col": float(
                (cols * weights_at_mask).sum()
                / weights_at_mask.sum()
            ),
        }

    else:
        bounding_box = None
        centroid = None

    return {
        "heatmap": cam_resized,
        "target_class_index": target_class_index,
        "target_class_name": ID2LABEL[target_class_index],
        "target_layer_name": target_layer_name,
        "bounding_box": bounding_box,
        "centroid": centroid,
    }


def save_gradcam_visualization(
    pixel_array,
    rescale_slope,
    rescale_intercept,
    gradcam_result,
    output_path,
):
    hu = to_hounsfield(
        pixel_array,
        rescale_slope,
        rescale_intercept,
    )

    brain_view = apply_window(
        hu,
        *BRAIN_WINDOW,
    )

    heatmap = gradcam_result["heatmap"]
    bounding_box = gradcam_result["bounding_box"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(
        brain_view,
        cmap="gray",
    )
    axes[0].set_title(
        "Original slice (brain window)"
    )
    axes[0].axis("off")

    axes[1].imshow(
        heatmap,
        cmap="jet",
    )
    axes[1].set_title(
        f"Grad-CAM: {gradcam_result['target_class_name']}"
    )
    axes[1].axis("off")

    axes[2].imshow(
        brain_view,
        cmap="gray",
    )
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
    )

    if bounding_box is not None:
        rect = plt.Rectangle(
            (
                bounding_box["col_min"],
                bounding_box["row_min"],
            ),
            bounding_box["col_max"]
            - bounding_box["col_min"],
            bounding_box["row_max"]
            - bounding_box["row_min"],
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )

        axes[2].add_patch(rect)

    axes[2].set_title(
        "Overlay + localized region"
    )
    axes[2].axis("off")

    fig.tight_layout()

    output_path = str(output_path)
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)

    return output_path