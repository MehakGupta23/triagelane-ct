# AuraLane — CT Hemorrhage Detection Lane

A CT-based intracranial hemorrhage detection and urgency-triage pipeline for
the AuraLane system.

The pipeline processes a DICOM CT study, estimates per-slice hemorrhage
likelihood, aggregates slice-level predictions into a study-level score, and
assigns the study to an urgency lane with an abstention mechanism for
borderline or potentially mismatched cases.

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
9. Abstention for boundary/mismatch cases

The final output contains the study-level score, raw score, dominant
hemorrhage subtype, urgency lane, and abstention status.

---

## 2. Pipeline Architecture

```text
                DICOM Study
                     │
                     ▼
          ┌─────────────────────┐
          │     DICOM I/O       │
          │ discovery + quality │
          │       checks        │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   Preprocessing     │
          │                     │
          │ Pixel → HU          │
          │ Brain window        │
          │ Subdural window     │
          │ Bone window         │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   Slice Model       │
          │                     │
          │ Image Processor     │
          │ Classification      │
          │ Temperature         │
          │ Calibration         │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │    Aggregation      │
          │                     │
          │ Top-k slice scores  │
          │ Study-level score   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │    Lane Assignment  │
          │                     │
          │ Critical            │
          │ Urgent              │
          │ Expedited           │
          │ Routine             │
          │ Abstention          │
          └──────────┬──────────┘
                     │
                     ▼
               result.json