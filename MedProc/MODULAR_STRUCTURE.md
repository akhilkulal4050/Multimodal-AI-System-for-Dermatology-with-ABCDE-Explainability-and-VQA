# MedProc Module Separation & Reorganization

## Overview

The MedProc (Medical Information Processing) component has been reorganized into **modular, reusable Python modules and well-structured Jupyter notebooks** for better maintainability, testability, and scalability.

## New File Structure

```
/MedProc/
├── medproc_dataset.py               [Core dataset utilities - REUSABLE MODULE]
├── medproc_model.py                 [Model architecture - REUSABLE MODULE]
├── medproc_train.ipynb              [Training execution - 6 SECTIONS]
├── medproc_visualize.ipynb          [Progress dashboard - 5 SECTIONS]
├── medproc_full_pipeline.ipynb      [End-to-end inference - UPDATED]
├── requirements.txt                 [Python dependencies]
└── README.md                        [This file]
```

---

## Module Descriptions

### 1. **medproc_dataset.py** — Dataset Utilities (Reusable)

**Purpose:** Handle all data loading, preprocessing, cleaning, and filtering operations.

**Key Functions:**
- `clean_text(text)` - Removes PHI tags, normalizes whitespace
- `smart_truncate(text, max_len)` - Prefers Assessment/Plan section when truncating
- `extract_symptoms_from_text(text)` - Rule-based keyword extraction
- `filter_icd_column(df, icd_col)` - Keeps only whitelisted ICD codes
- `icd_is_relevant(code)` - Checks if ICD code is in whitelist
- `save_with_progress(ddf, path)` - Save Dask DataFrame with progress bar

**Key Constants:**
- `ICD_WHITELIST` - 200+ whitelisted ICD-9 codes (skin + general)
- `SYMPTOM_KEYWORDS` - 60+ clinical keywords
- `MAX_TEXT_LENGTH` - 3000 characters max per note
- `OUTPUT_DIR`, `DATA_ROOT` - Path configurations

**Usage:**
```python
from medproc_dataset import clean_text, extract_symptoms_from_text, ICD_WHITELIST

# Clean a clinical note
clean_note = clean_text(raw_note)

# Extract symptoms
symptoms = extract_symptoms_from_text(clean_note)

# Check if ICD code is relevant
is_relevant = ICD_WHITELIST.intersection({'172'})  # melanoma
```

---

### 2. **medproc_model.py** — Model Architecture & Training (Reusable)

**Purpose:** Define the multi-task learning model and provide training/evaluation functions.

**Key Classes:**
- `MedProcMultiTaskModel` - Bio_ClinicalBERT backbone with 2 task heads:
  - Head 1: ICD-9 multi-label classification (176+ classes)
  - Head 2: Symptom relevance detection (binary)
- `MedProcDataset` - PyTorch Dataset for tokenization and batching

**Key Functions:**
- `create_dataloaders()` - Create train/val DataLoaders
- `train_epoch(model, loader, ...)` - Single training iteration
- `eval_epoch(model, loader, ...)` - Validation evaluation
- `save_checkpoint(model, tokenizer, ...)` - Save complete checkpoint
- `load_checkpoint(path)` - Restore from checkpoint

**Key Constants:**
- `EPOCHS = 50`, `LR = 2e-5`, `BATCH_SIZE = 16`
- `WEIGHT_ICD = 1.0`, `WEIGHT_SYMPTOM = 0.5`
- `MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'`

**Usage:**
```python
from medproc_model import MedProcMultiTaskModel, create_dataloaders, train_epoch

# Initialize model
model = MedProcMultiTaskModel(model_name='emilyalsentzer/Bio_ClinicalBERT', 
                             num_icd_labels=176)

# Create dataloaders
train_dl, val_dl = create_dataloaders(texts, icd_labels, symptom_labels, 
                                      train_idx, val_idx, tokenizer)

# Train one epoch
train_loss = train_epoch(model, train_dl, optimizer, scheduler, ...)
```

---

### 3. **medproc_train.ipynb** — Training Execution (New)

**Purpose:** Execute model training with clear separation of concerns into 6 sections.

**Sections:**
1. **Load and Prepare Dataset** - Load parquet/demo data, clean, split
2. **Define Model Architecture** - Initialize tokenizer and model
3. **Train Model** - 50 epochs with early stopping
4. **Evaluate Model Performance** - Compute ICD accuracy, precision, recall, F1, symptom accuracy
5. **Visualize Training Progress** - Loss curves and accuracy plots
6. **Visualize Predictions** - Sample predictions with confidence scores

**Outputs:**
- `training_progress.png` - Loss and accuracy curves
- `symptom_confusion_matrix.png` - Binary classification matrix
- `medproc_best.pt` - Best model checkpoint
- `medproc_checkpoint.pt` - Complete checkpoint with tokenizer

---

### 4. **medproc_visualize.ipynb** — Progress Dashboard (New)

**Purpose:** Track project status, visualize metrics, and monitor integration progress.

**Sections:**
1. **Project Status & Components** - File inventory and component status
2. **Model Architecture & Capacity** - Architecture diagram and parameter summary
3. **Expected Performance Metrics** - Target benchmarks for all tasks
4. **Project Roadmap & Integration Plan** - Timeline from Phase 1 (Module sep.) to Phase 5 (Production)
5. **Summary & Quick Reference** - Usage guide and next steps

**Outputs:**
- `model_architecture.png` - Architecture visualization
- `performance_targets.png` - Benchmark dashboard
- `project_roadmap.png` - Project timeline

---

### 5. **medproc_full_pipeline.ipynb** — End-to-End Pipeline (Updated)

**Changes:**
- Fixed Cell 20 (Model Evaluation) - Corrected array concatenation
- Fixed Cell 21 (Inference Pipeline) - JSON serialization for numpy types
- Fixed Cell 23 (Batch Demo) - Complete batch prediction with CSV export

**Purpose:** End-to-end inference pipeline with trained model.

**Key Functions:**
- `infer_medproc_single()` - Single prediction with confidence scores
- `infer_medproc_batch()` - Batch predictions on multiple notes
- `predict_batch_and_export()` - Batch predict and save to CSV

---

## Workflow

### Phase 1: Development & Training ✅ COMPLETED

```bash
# 1. Prepare environment
cd /home/vjti-comp/Desktop/Final\ Project\ Code/MedProc
python -m pip install -r requirements.txt

# 2. Run training notebook
jupyter notebook medproc_train.ipynb
# → Executes training (50 epochs), saves best model

# 3. Check progress
jupyter notebook medproc_visualize.ipynb
# → View metrics, benchmarks, roadmap
```

### Phase 2: Inference & Prediction

```bash
# 4. Run inference pipeline
jupyter notebook medproc_full_pipeline.ipynb
# → Load trained model, run batch predictions, export results
```

### Phase 3: Integration (Next Steps)

```bash
# 5. Integrate with HC module
cd /home/vjti-comp/Desktop/Final\ Project\ Code/HC
# → Process outputs from MedProc as input to HC classifier

# 6. Package as REST API for production
# → Wrap modules in Flask/FastAPI endpoints
```

---

## Expected Performance

Target metrics from `medproc_visualize.ipynb`:

| Task | Metric | Target |
|------|--------|--------|
| **ICD Classification** | Exact Match Accuracy | >75% |
| | Per-class F1 | >0.65 |
| | ROC-AUC (top-5) | >0.85 |
| **Symptom Detection** | Accuracy | >80% |
| | Precision/Recall | >0.75 |
| | F1-Score | >0.77 |
| **Symptom Extraction** | Keyword Match Rate | >70% |
| **Evolution Signal** | Detection Sensitivity | >75% |

---

## File Dependencies

```
medproc_train.ipynb
├── imports: medproc_dataset.py
├── imports: medproc_model.py
└── outputs: training_progress.png, symptom_confusion_matrix.png

medproc_visualize.ipynb
├── reads: training artifacts from medproc_train.ipynb
└── outputs: model_architecture.png, performance_targets.png, project_roadmap.png

medproc_full_pipeline.ipynb
├── imports: medproc_model.py
├── loads: medproc_checkpoint.pt (from medproc_train.ipynb)
└── outputs: medproc_predictions.csv
```

---

## Key Improvements

✅ **Modularity** - Reusable dataset and model utilities for other projects  
✅ **Clarity** - 6-section training notebook is easy to understand and modify  
✅ **Traceability** - Progress dashboard tracks metrics and milestones  
✅ **Reproducibility** - All hyperparameters and configs centralized  
✅ **Scalability** - Easy to extend with new heads, tasks, or datasets  

---

## Next Steps

1. **Data Acquisition** - Obtain MIMIC-III/IV access (credentialing required)
2. **Model Optimization** - Fine-tune hyperparameters using `medproc_train.ipynb`
3. **HC Integration** - Connect MedProc output → HC hierarchical classifier input
4. **Full Pipeline** - Package both modules for IT-Fusion production deployment
5. **Validation** - Test on hold-out evaluation set with real medical data

---

## Contact & Support

For questions on MedProc architecture, training, or integration, refer to:
- `medproc_visualize.ipynb` Section 5 for quick reference
- Individual module docstrings in `.py` files
- Comments in Jupyter notebook cells

**Date Created:** April 6, 2026  
**Project:** NutriDermAI - Stage 5 (Medical Information Processing)
