# IT Fusion Modular Integration Guide (Stage 6)

This guide documents the modular, clinical integration layer (Stage 6) of the Multi-Modal Dermatology Diagnostic System.

## Architecture & Module Directory

The active pipeline consists of the following modular files in `it_fusion_modular_project/`:

*   **`fusion.py`**: The master orchestrator. Loads inputs (segmentation mask, cropped ROI image, patient history/text features, demographics), triggers the ABCDE feature extraction, forwards all 5 modalities to the `MARIADermatology` transformer, and coordinates the generation of the final clinical report and JSON.
*   **`abcde_computation.py`**: Computes the official morphological ABCDE parameters (Asymmetry, Border, Color, Diameter, Evolution) directly from the **Stage 2 segmentation mask** and **Stage 3 cropped ROI image** using computer vision algorithms (moments, contour analysis, box-counting fractal dimension, color variance).
*   **`abcde_inference.py`**: Interprets the raw calculated ABCDE scores into a risk category, and formats the feature vectors for the MARIA transformer.
*   **`json_builder.py`**: Constructs the final structured, clinical JSON containing classification, risk levels, detailed ABCDE values, and clinical recommendations.
*   **`explainability_generator.py`**: Generates clinical descriptions and feature attribution mapping for transparency and validation.
*   **`drug_rules.py`**: Rule-based treatment recommender that recommends therapies based on classification, risk level, and diagnostic metadata. *(Note: This is a Stage 6 clinical knowledge base, NOT part of Stage 5 MedProc.)*

---

## Data Flow Pipeline

```
┌─────────────────────────────────┐
│ Stage 2 Segmentation Mask       │
│ Stage 3 Cropped ROI Image        ├─┐
└─────────────────────────────────┘ │
                                    ▼
                     ┌─────────────────────────────┐
                     │    abcde_computation.py     │
                     │  Computes A, B, C, D scores │
                     └──────────────┬──────────────┘
                                    │
┌─────────────────────────────────┐ │
│ Stage 5 MedProc                 │ │
│ • Evolution score (E)           ├─┼─► Evolution (E) score
│ • Symptom keywords              ├─┼─► Symptoms modality (6-dim)
│ • ICD classification            │ │   [itch, grew, hurt, changed,
│ (No drug detection — removed)   │ │    bleed, elevation]
└─────────────────────────────────┘ │
                                    ▼
                     ┌─────────────────────────────┐
                     │     abcde_inference.py      │
                     │  Translates to risk vector  │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
┌─────────────────────────────────┐  Modality 1: Demographics
│ Demographics, Symptoms, History ├─► Modality 2: Symptoms (from MedProc keywords)
│ & Image Metadata                │  Modality 3: Metadata (from HC features)
└─────────────────────────────────┘  Modality 4: Medical History
                                     Modality 5: ABCDE Feature Vector (13-dim)
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │          fusion.py          │
                     │  (MARIADermatology Model)   │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │       json_builder.py       │
                     │  (Unified Clinical JSON)    │
                     └─────────────────────────────┘
```

## Stage 5 → Stage 6 Input Mapping

| Stage 5 (MedProc) Output | Stage 6 Consumer | How |
|---------------------------|-----------------|-----|
| `evolution_score` (float 0–1) | `abcde_computation.compute_evolution()` | Directly used as base E score |
| `symptom_keywords` (list) | `fusion.py` → symptoms modality (6-dim) | Keyword-mapped to binary vector |
| `symptom_keywords` (list) | `abcde_computation.compute_evolution()` | Keyword boosting for E score |
| Bio_ClinicalBERT embeddings | `dataset.ipynb` → text features (768-dim) | Used for MARIA cross-attention |

**Note**: Drug detection was removed from MedProc. Treatment recommendations are handled by `drug_rules.py` (a rule-based clinical knowledge base in Stage 6).

## How to Run the Pipeline

The pipeline is run by invoking `run_it_fusion_pipeline` inside `fusion.py`:

```python
from fusion import run_it_fusion_pipeline

result = run_it_fusion_pipeline(
    image_path="path/to/roi_crop.png",
    mask_path="path/to/segmentation_mask.png",
    demographics={"age": 45, "gender": "male"},
    symptoms=["bleeding", "itching"],
    history="Family history of melanoma, lesion has grown over 3 months.",
    metadata={"body_location": "back"}
)

# Output is a unified Clinical JSON ready for Stage 7 VQA Layer
print(result)
```
