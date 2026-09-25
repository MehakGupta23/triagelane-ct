import torch
import pydicom
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from .config import CONFIG
from .dicom_io import validate_dicom_slice
from .preprocessing import three_window_stack


MODEL_NAME = CONFIG["model"]["name"]
MODEL_REVISION = CONFIG["model"].get("revision")

TEMPERATURE = CONFIG["calibration"]["temperature"]

URGENCY_WEIGHTS = CONFIG["urgency_weights"]

ID2LABEL = {
    int(k): v
    for k, v in CONFIG["id2label"].items()
}


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


_extractor = None
_model = None


class ModelLabelMismatchError(ValueError):
    pass


def _verify_id2label(model):
    model_id2label = getattr(
        model.config,
        "id2label",
        None,
    )

    if not model_id2label:
        return

    normalized_model = {
        int(k): str(v).strip().lower()
        for k, v in model_id2label.items()
    }

    normalized_expected = {
        k: v.lower()
        for k, v in ID2LABEL.items()
    }

    if normalized_model != normalized_expected:
        raise ModelLabelMismatchError(
            f"{normalized_model} != {normalized_expected}"
        )


def load_model():
    global _extractor, _model

    if _model is None:

        load_kwargs = {}

        if MODEL_REVISION:
            load_kwargs["revision"] = MODEL_REVISION

        extractor = AutoImageProcessor.from_pretrained(
            MODEL_NAME,
            **load_kwargs,
        )

        model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            **load_kwargs,
        )

        _verify_id2label(model)

        model.to(DEVICE)
        model.eval()

        _extractor = extractor
        _model = model

    return _extractor, _model


def predict_slice(
    pixel_array,
    rescale_slope,
    rescale_intercept,
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

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    normal_logit = logits[3]

    abnormal_logit = torch.logsumexp(
        torch.cat([logits[:3], logits[4:]]),
        dim=0,
    )

    calibrated = torch.softmax(
        torch.stack([
            abnormal_logit,
            normal_logit,
        ]) / TEMPERATURE,
        dim=0,
    )

    calibrated_likelihood = calibrated[0].item()

    calibrated_probs = torch.softmax(
        logits / TEMPERATURE,
        dim=0,
    )

    subtype_probs = {
        "epidural": calibrated_probs[0].item(),
        "intraparenchymal": calibrated_probs[1].item(),
        "intraventricular": calibrated_probs[2].item(),
        "subarachnoid": calibrated_probs[4].item(),
        "subdural": calibrated_probs[5].item(),
    }

    dominant_subtype = max(
        subtype_probs,
        key=subtype_probs.get,
    )

    urgency_weighted_score = min(
        calibrated_likelihood
        * URGENCY_WEIGHTS[dominant_subtype],
        1.0,
    )

    return {
        "calibrated_likelihood": calibrated_likelihood,
        "subtype_probs": subtype_probs,
        "dominant_subtype": dominant_subtype,
        "urgency_weighted_score": urgency_weighted_score,
    }


def predict_slice_from_dicom_path(path):
    dcm = pydicom.dcmread(path)

    pixel_array = validate_dicom_slice(dcm)

    return predict_slice(
        pixel_array,
        dcm.RescaleSlope,
        dcm.RescaleIntercept,
    )