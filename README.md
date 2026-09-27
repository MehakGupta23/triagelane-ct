# AuraLane — CT Hemorrhage Detection Lane

A CT-based intracranial hemorrhage detection and urgency-triage pipeline for the AuraLane system.

The pipeline processes a DICOM CT study, estimates per-slice hemorrhage likelihood, aggregates slice-level predictions into a study-level score, assigns the study to an urgency lane, and optionally generates Grad-CAM localization for the highest-scoring slice.

---

## 1. Project Overview

The pipeline performs the following stages:

1. DICOM study discovery and validation
2. CT preprocessing and Hounsfield Unit conversion
3. Multi-window CT image generation
4. Per-slice hemorrhage classification
5. Temperature calibration
6. Urgency-weighted scoring
7. Study-level top-k aggregation
8. Urgency-lane assignment
9. Abstention for boundary and mismatch cases
10. Optional Grad-CAM localization

The pipeline is organized into separate modules so that DICOM handling, preprocessing, model inference, aggregation, lane assignment, and localization remain independently testable.

---

## 2. Pipeline Architecture

```text
DICOM CT Study
      │
      ▼
DICOM Discovery & Validation
      │
      ▼
Pixel Array + Rescale Parameters
      │
      ▼
Hounsfield Unit Conversion
      │
      ▼
Multi-Window CT Preprocessing
      │
      ├── Brain Window
      ├── Subdural Window
      └── Bone Window
      │
      ▼
Per-Slice Model Inference
      │
      ▼
Temperature Calibration
      │
      ▼
Per-Slice Probabilities
      │
      ▼
Urgency-Weighted Slice Scores
      │
      ▼
Top-k Slice Selection
      │
      ▼
Study-Level Aggregation
      │
      ▼
Urgency-Lane Assignment
      │
      ├── Critical
      ├── Urgent
      ├── Expedited
      └── Routine
      │
      ▼
Boundary / Mismatch Abstention
      │
      ▼
Final Study Result
      │
      └── Optional Grad-CAM
              │
              ▼
        Top-Scoring Slice
              │
              ▼
        Heatmap + Bounding Box
              │
              ▼
        PNG Visualization
```

Grad-CAM is an optional localization step. It runs after the lane decision and does not modify the study score or lane assignment.

---

## 3. Model

The pipeline uses a transformer-based image classification model for per-slice hemorrhage classification.

The model predicts the following classes:

| Class Index | Class |
|-------------:|-------|
| 0 | Epidural |
| 1 | Intraparenchymal |
| 2 | Intraventricular |
| 3 | Normal |
| 4 | Subarachnoid |
| 5 | Subdural |

The model is loaded once and reused across slice inference.

Model loading and inference are implemented in:

`src/triagelane_ct/model.py`

---

## 4. CT Preprocessing

Each DICOM slice is converted into Hounsfield Units using the DICOM rescale parameters:

`HU = PixelArray × RescaleSlope + RescaleIntercept`

The pipeline then creates multiple CT window representations.

Current configured windows:

| Window | Center | Width |
|--------|-------:|------:|
| Brain | 40 | 80 |
| Subdural | 100 | 200 |
| Bone | 600 | 2800 |

The preprocessing implementation is located in:

`src/triagelane_ct/preprocessing.py`

The resulting multi-window representation is passed to the classification model.

---

## 5. Per-Slice Inference

For every valid DICOM slice, the pipeline performs:

1. DICOM pixel extraction
2. Hounsfield Unit conversion
3. CT window generation
4. Model inference
5. Temperature calibration
6. Class probability generation
7. Urgency-weighted scoring
8. Dominant subtype selection

The inference logic is implemented in:

`src/triagelane_ct/model.py`

The resulting slice-level predictions are passed to the study-level aggregation stage.

---

## 6. Temperature Calibration

Raw model outputs are calibrated before being used for study-level scoring.

Temperature calibration separates the model's raw classification output from the probabilities used by the triage logic.

The calibrated probabilities are then used to calculate the urgency-weighted score for every slice.

---

## 7. Urgency-Weighted Scoring

Different hemorrhage subtypes contribute differently to the urgency score.

The urgency weights are configured in:

`configs/ct_pipeline.yaml`

The urgency-weighted score is calculated from the calibrated class probabilities and the configured subtype weights.

This score is used for:

- selecting the highest-scoring slices
- study-level aggregation
- urgency lane assignment
- selecting the slice used for optional Grad-CAM localization

---

## 8. Study-Level Aggregation

A CT study contains multiple slices. The pipeline aggregates slice-level predictions into a single study-level score.

The aggregation stage selects the highest-scoring slices rather than treating every slice equally.

The aggregation logic is implemented in:

`src/triagelane_ct/aggregation.py`

The resulting study-level values include:

- `study_score`
- `raw_score`
- `k_used`
- `n_slices`
- `dominant_subtype`

The `study_score` is the urgency-weighted study score used for lane assignment.

---

## 9. Urgency Lane Assignment

The study-level urgency score is mapped to one of four urgency lanes.

Current thresholds:

| Lane | Threshold |
|------|----------:|
| Critical | ≥ 0.85 |
| Urgent | ≥ 0.65 |
| Expedited | ≥ 0.35 |
| Routine | < 0.35 |

The lane assignment logic is implemented in:

`src/triagelane_ct/lanes.py`

The lane module also handles boundary conditions and mismatch checks.

---

## 10. Abstention Logic

The pipeline does not force a lane assignment for cases that fall into configured uncertainty or mismatch conditions.

Abstention can occur around lane boundaries or when the predicted subtype and urgency characteristics produce a configured mismatch.

When abstention occurs, the result contains:

- `abstain`
- `abstain_reason`
- `in_abstention_tray`

When the study is not abstained, the result contains the assigned lane and decision reason.

---

## 11. Decision Reason

Every inference result contains a `decision_reason`.

For a normal lane assignment, the reason describes how the urgency-weighted study score relates to the configured lane thresholds and identifies the dominant subtype contributing to the decision.

For an abstained case, the configured abstention reason is returned instead.

This provides an explicit explanation of the rule-based triage decision without changing the underlying model prediction.

---

## 12. Grad-CAM Localization

Grad-CAM is an optional post-processing stage used to visualize where the model's prediction is concentrated.

It is enabled using the `--localize` command-line flag.

The localization flow is:

```text
Study-Level Inference
        │
        ▼
Lane Decision
        │
        ▼
Highest Urgency-Weighted Slice
        │
        ▼
Dominant Hemorrhage Subtype
        │
        ▼
Grad-CAM Target Class
        │
        ▼
Transformer Target Layer
        │
        ▼
Grad-CAM Heatmap
        │
        ▼
Localization Threshold
        │
        ▼
Bounding Box + Centroid
        │
        ▼
PNG Visualization
```

The Grad-CAM implementation is located in:

`src/triagelane_ct/gradcam.py`

The configured localization threshold is:

`0.5`

The threshold is configured in:

`configs/ct_pipeline.yaml`

Grad-CAM does not feed back into the lane assignment or study score.

---

## 13. Grad-CAM Target Selection

Grad-CAM is generated for the dominant hemorrhage subtype of the highest-scoring slice.

The pipeline determines the target class from the dominant subtype and locates a suitable transformer layer for Grad-CAM.

The resulting localization contains:

- selected slice index
- selected slice ID
- target class name
- target transformer layer
- bounding box
- centroid
- visualization path

The localization is therefore tied directly to the slice that contributed most strongly to the study-level urgency score.

---

## 14. Grad-CAM Visualization

The generated visualization contains three panels:

1. Original CT slice
2. Raw Grad-CAM heatmap
3. Grad-CAM overlay

The visualization is saved as a PNG file.

Example output structure:

`outputs/<study>/gradcam/gradcam_slice_<n>.png`

The heatmap is a model-derived localization and should be interpreted as a visualization of model attention rather than as a segmentation mask.

---

## 15. Output Schema

A normal inference result contains fields such as:

- `study_score`
- `raw_score`
- `k_used`
- `n_slices`
- `dominant_subtype`
- `decision_reason`
- `lane`
- `abstain`
- `in_abstention_tray`
- `sort_score`
- `abstain_reason`

When Grad-CAM localization is enabled, an additional `localization` object is included.

The localization object contains:

- `slice_index`
- `slice_id`
- `target_class_name`
- `target_layer`
- `bounding_box`
- `centroid`
- `visualization_path`

Example structure:

```json
{
  "study_score": 0.45,
  "raw_score": 0.412,
  "k_used": 6,
  "n_slices": 19,
  "dominant_subtype": "subarachnoid",
  "decision_reason": "...",
  "lane": "expedited",
  "abstain": false,
  "in_abstention_tray": false,
  "sort_score": "...",
  "abstain_reason": null,
  "localization": {
    "slice_index": 5,
    "slice_id": "...",
    "target_class_name": "subarachnoid",
    "target_layer": "...",
    "bounding_box": {
      "row_min": 149,
      "row_max": 431,
      "col_min": 185,
      "col_max": 383
    },
    "centroid": {
      "row": 357.05,
      "col": 267.88
    },
    "visualization_path": "gradcam/gradcam_slice_5.png"
  }
}
```

---

## 16. Running the Pipeline

The project provides a command-line inference script:

`scripts/run_inference.py`

### Normal inference

Run inference on a DICOM study without localization:

`python scripts/run_inference.py --input data/demo/ID_8db4b6544e --output outputs/study_001`

This generates the standard inference output without Grad-CAM.

### Inference with Grad-CAM

To additionally generate localization:

`python scripts/run_inference.py --input data/demo/ID_8db4b6544e --output outputs/study_001_localized --localize`

The localized run creates:

`outputs/study_001_localized/gradcam/gradcam_slice_5.png`

along with the normal inference result.

---

## 17. Configuration

Pipeline configuration is centralized in:

`configs/ct_pipeline.yaml`

The configuration contains parameters for:

- model configuration
- class labels
- CT windows
- aggregation
- urgency weights
- lane thresholds
- boundary margin
- mismatch checks
- Grad-CAM localization threshold

Example Grad-CAM configuration:

```yaml
gradcam:
  localization_threshold: 0.5
```

Keeping these parameters in configuration rather than hard-coding them inside the pipeline makes the inference behavior easier to inspect and modify.

---

## 18. Project Structure

```text
ct-lane/
├── .venv/                         # Local virtual environment; not committed
│
├── configs/
│   └── ct_pipeline.yaml           # Pipeline configuration
│
├── data/                          # Local data; not committed
│   └── demo/                      # Demo DICOM studies
│
├── outputs/                       # Local inference outputs; not committed
│
├── scripts/
│   └── run_inference.py           # CLI entry point
│
├── src/
│   └── triagelane_ct/
│       ├── __init__.py
│       ├── aggregation.py         # Study-level score aggregation
│       ├── config.py              # Configuration loading
│       ├── dicom_io.py            # DICOM discovery and validation
│       ├── gradcam.py             # Grad-CAM localization
│       ├── lanes.py               # Lane assignment and abstention
│       ├── model.py               # Model loading and inference
│       ├── pipeline.py            # Main inference orchestration
│       └── preprocessing.py       # CT preprocessing
│
├── tests/
│   ├── __init__.py
│   ├── test_aggregation.py
│   ├── test_dicom_io.py
│   ├── test_gradcam.py
│   ├── test_lanes.py
│   └── test_preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 19. Module Responsibilities

### `config.py`

Loads the YAML configuration and exposes the configuration to the rest of the pipeline.

### `dicom_io.py`

Responsible for:

- DICOM discovery
- DICOM slice validation
- study folder handling
- DICOM-related input validation

### `preprocessing.py`

Responsible for:

- Hounsfield Unit conversion
- CT windowing
- model input preparation

### `model.py`

Responsible for:

- model loading
- model inference
- per-slice prediction
- probability calibration

### `aggregation.py`

Responsible for:

- urgency-weighted slice scoring
- top-k selection
- study-level aggregation

### `lanes.py`

Responsible for:

- urgency lane assignment
- threshold handling
- boundary checks
- mismatch checks
- abstention logic
- lane decision reason

### `gradcam.py`

Responsible for:

- Grad-CAM target class selection
- transformer target-layer discovery
- activation reshaping
- Grad-CAM computation
- localization thresholding
- bounding box extraction
- centroid calculation
- visualization generation

### `pipeline.py`

Acts as the main orchestration layer.

It connects:

```text
DICOM
  ↓
Preprocessing
  ↓
Model
  ↓
Aggregation
  ↓
Lane Assignment
  ↓
Optional Grad-CAM
  ↓
Final Result
```

### `scripts/run_inference.py`

Provides the command-line interface for running inference on a DICOM study.

---

## 20. Testing

The project uses `pytest` for automated testing.

Run the complete test suite with:

`pytest -q`

The test suite covers:

- preprocessing behavior
- DICOM validation
- aggregation
- lane assignment
- Grad-CAM localization integration

The current test suite contains 12 tests, all passing.

Grad-CAM testing verifies that the localization output contains:

- selected slice index
- slice ID
- target class
- target layer
- bounding box
- centroid
- visualization path

---

## 21. Demo Studies

Three prototype DICOM studies are currently used for local validation:

- `ID_8db4b6544e`
- `ID_7558cf1f54`
- `ID_a79b6e570b`

Demo studies are stored locally under:

`data/demo/`

The DICOM data is excluded from Git through `.gitignore`.

---

## 22. Example Validation

A normal local inference run has been validated on the prototype study:

`ID_8db4b6544e`

Observed study-level output:

- Study score: `0.45`
- Raw score: `0.412`
- Top-k slices used: `6`
- Total slices: `19`
- Dominant subtype: `subarachnoid`
- Lane: `expedited`
- Abstention: `false`

A localized inference run on the same study successfully generated:

`gradcam/gradcam_slice_5.png`

The generated Grad-CAM localization included:

- target class: `subarachnoid`
- slice index: `5`
- bounding box
- centroid
- PNG visualization

---

## 23. Data and Output Handling

Input DICOM studies are kept outside version control.

Local data is stored under:

`data/`

Inference outputs are stored under:

`outputs/`

The virtual environment is stored under:

`.venv/`

These local artifacts are excluded from Git.

The repository is intended to contain source code, configuration, tests, documentation, and dependency definitions rather than patient data, generated outputs, or local environments.

---

## 24. Dependencies

The main runtime dependencies include:

- Python 3.12
- PyTorch
- Transformers
- NumPy
- pydicom
- Pillow
- PyYAML
- Matplotlib

Testing uses:

- pytest

The complete pinned environment is recorded in:

`requirements.txt`

For the current NVIDIA environment, PyTorch is installed with CUDA 12.6 support.

---

## 25. Local Environment Setup

Create the Python virtual environment:

`py -3.12 -m venv .venv`

Activate it on Windows:

`.venv\Scripts\activate`

Upgrade pip:

`python -m pip install --upgrade pip`

Install the project dependencies:

`pip install -r requirements.txt`

After installation, verify the test suite:

`pytest -q`

---

## 26. Inference Workflow

The recommended local workflow is:

1. Place a DICOM study under `data/demo/` or another local input directory.
2. Activate the Python virtual environment.
3. Run normal inference using `scripts/run_inference.py`.
4. Inspect the generated inference result.
5. Run the same study with `--localize` when localization is required.
6. Inspect the generated Grad-CAM PNG.
7. Run `pytest -q` after code changes.

---

## 27. Design Notes

The pipeline intentionally separates classification from triage.

The model produces slice-level predictions.

The aggregation layer converts those predictions into a study-level score.

The lane layer applies the configured urgency thresholds and abstention rules.

Grad-CAM is then used as an optional explanatory localization layer on the highest-scoring slice.

Therefore:

```text
Classification
      ↓
Scoring
      ↓
Aggregation
      ↓
Lane Decision
      ↓
Optional Localization
```

Grad-CAM is not used to determine the lane.

---

## 28. Current Validation Status

The local pipeline has been validated through:

- DICOM discovery and validation
- model loading
- normal study inference
- study-level aggregation
- urgency lane assignment
- decision reason generation
- Grad-CAM generation
- Grad-CAM localization output
- automated unit/integration tests

Current automated test status:

`12 passed`

The prototype local inference has also successfully generated a Grad-CAM visualization for a real demo DICOM study.

---

## 29. Repository Scope

The Git repository contains the reusable inference pipeline and its supporting configuration and tests.

The following are intentionally kept local and are not committed:

- DICOM studies
- generated inference outputs
- Grad-CAM PNGs
- virtual environments
- model checkpoints
- other local runtime artifacts

This keeps the repository focused on the reproducible pipeline implementation rather than local data or generated artifacts.