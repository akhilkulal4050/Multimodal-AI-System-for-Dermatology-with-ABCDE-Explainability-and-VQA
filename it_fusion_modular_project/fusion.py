"""
fusion.py
─────────
IT Fusion Stage (Stage 6) — the centrepiece of the system.

Orchestrates:
  1. Receives inputs from HC (Stage 4) and MedProc (Stage 5)
  2. Computes ABCDE from mask + ROI + evolution score
  3. Builds MARIA-format modality tensors
  4. Runs MARIA Transformer for cross-modal fusion
  5. Produces unified Clinical JSON

MARIA here operates on 5 modalities:
  X1 – Demographics       (from dataset preprocessing)
  X2 – Symptoms           (from dataset preprocessing + MedProc)
  X3 – Image metadata     (from dataset preprocessing + HC features)
  X4 – Medical history    (from dataset preprocessing)
  X5 – ABCDE features     (computed here, NEW modality)

The MARIA model code below is an adaptation of the paper's architecture
(Caruso et al., arXiv:2412.14810v2) to the 5-modality dermatology setting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import json
import os
import sys

# ── add local modules to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../abcde'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../drug_rules'))

from abcde_computation import compute_abcde, ABCDEResult
from abcde_inference   import interpret_abcde, abcde_to_feature_vector, ClinicalReport
from drug_rules        import recommend_treatment


# ═════════════════════════════════════════════════════════════════════════════
# 1.  NAIM-STYLE TABULAR ENCODER (modality-specific)
#     Implements the masked self-attention from MARIA (Eq. 1–3 of paper)
# ═════════════════════════════════════════════════════════════════════════════

class MaskedMultiHeadAttention(nn.Module):
    """
    Modified multi-head attention that completely ignores missing tokens.
    Masking follows Eq. (2)–(3) from MARIA paper:
      MSA = ReLU(softmax(QKᵀ/√dₕ + M + Mᵀ)) V
    where M_kj = -∞ if feature j is missing, else 0.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_h     = d_model // n_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, T, d_model)   token embeddings
        mask : (B, T)             binary — 1 = present, 0 = missing

        Returns: (B, T, d_model)
        """
        B, T, _ = x.shape

        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_h).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_h).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_h).transpose(1, 2)
        # (B, n_heads, T, d_h)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_h ** 0.5)
        # (B, n_heads, T, T)

        # Build additive mask: −∞ for missing column j
        # mask shape: (B, T) → expand to (B, 1, 1, T) for broadcasting
        additive = (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * (-1e9)
        # Also mask rows (keys of missing tokens shouldn't query others)
        additive_row = (1.0 - mask.float()).unsqueeze(1).unsqueeze(3) * (-1e9)
        scores = scores + additive + additive_row  # M + Mᵀ from paper

        # ReLU(softmax(·)) — paper uses this instead of plain softmax
        attn = F.relu(F.softmax(scores, dim=-1))
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)               # (B, n_heads, T, d_h)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out)


class NAIMEncoder(nn.Module):
    """
    Modality-specific encoder Eᵢ from MARIA.
    Embeds each tabular feature as a token, then applies masked self-attention.
    """

    def __init__(self, n_features: int, d_embed: int = 32,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_embed    = d_embed

        # One learnable embedding per feature position
        self.token_embed = nn.Embedding(n_features, d_embed)
        # Value projection: scalar feature value → d_embed
        self.value_proj = nn.Linear(1, d_embed)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MaskedMultiHeadAttention(d_embed, n_heads, dropout),
                'norm1': nn.LayerNorm(d_embed),
                'ff':    nn.Sequential(
                    nn.Linear(d_embed, d_embed * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_embed * 4, d_embed),
                ),
                'norm2': nn.LayerNorm(d_embed),
            })
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, n_features)   raw feature values (0 for missing)
        mask : (B, n_features)   1 = present, 0 = missing

        Returns: rᵢ  shape (B, n_features, d_embed)
        """
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        # Token = positional embedding + value projection
        tok  = self.token_embed(pos).expand(B, -1, -1)             # (B, T, d)
        val  = self.value_proj(x.unsqueeze(-1))                     # (B, T, d)
        h    = tok + val * mask.unsqueeze(-1).float()               # zero-out missing

        for layer in self.layers:
            attn_out = layer['attn'](h, mask)
            h = layer['norm1'](h + attn_out)
            ff_out = layer['ff'](h)
            h = layer['norm2'](h + ff_out)

        # Zero-out representations of missing features
        h = h * mask.unsqueeze(-1).float()
        return h   # (B, T, d_embed)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  SHARED ENCODER  Eₛₕ  (cross-modal fusion)
# ═════════════════════════════════════════════════════════════════════════════

class SharedEncoder(nn.Module):
    """
    Shared encoder that fuses concatenated latent representations rₛₕ.
    Modality-level masking applied via modality_mask (B, n_modalities).
    """

    def __init__(self, total_tokens: int, d_embed: int = 32,
                 n_heads: int = 4, n_layers: int = 2,
                 n_classes: int = 7, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MaskedMultiHeadAttention(d_embed, n_heads, dropout),
                'norm1': nn.LayerNorm(d_embed),
                'ff':    nn.Sequential(
                    nn.Linear(d_embed, d_embed * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_embed * 4, d_embed),
                ),
                'norm2': nn.LayerNorm(d_embed),
            })
            for _ in range(n_layers)
        ])
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(d_embed, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, r_sh: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """
        r_sh       : (B, total_tokens, d_embed)  concatenated latent reps
        token_mask : (B, total_tokens)            1 = present, 0 = missing

        Returns: logits (B, n_classes)
        """
        h = r_sh
        for layer in self.layers:
            attn_out = layer['attn'](h, token_mask)
            h = layer['norm1'](h + attn_out)
            ff_out = layer['ff'](h)
            h = layer['norm2'](h + ff_out)

        # Pool over token dimension
        h = h * token_mask.unsqueeze(-1).float()
        h = h.transpose(1, 2)                          # (B, d, T)
        pooled = self.pool(h).squeeze(-1)              # (B, d)
        pooled = self.dropout(pooled)
        return self.head(pooled)                       # (B, n_classes)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  FULL MARIA MODEL FOR DERMATOLOGY
# ═════════════════════════════════════════════════════════════════════════════

# Modality definitions (must match dataset_prep/dataset_to_maria_format.ipynb)
MODALITY_SIZES = {
    'demographics'    : 3,
    'symptoms'        : 6,
    'image_meta'      : 5,
    'medical_history' : 8,
    'abcde'           : 13,  # from abcde_inference.abcde_to_feature_vector()
}
MODALITY_NAMES = list(MODALITY_SIZES.keys())
N_CLASSES = 7   # MEL, BCC, SCC, AKIEC, DF, VASC, NEV

CLASS_NAMES = ['MEL', 'BCC', 'SCC', 'AKIEC', 'DF', 'VASC', 'NEV']


class MARIADermatology(nn.Module):
    """
    Full MARIA model adapted for the 5-modality dermatology fusion task.
    """

    def __init__(self, d_embed: int = 64, n_heads: int = 4,
                 n_enc_layers: int = 2, n_shared_layers: int = 3,
                 n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.d_embed = d_embed

        # Modality-specific encoders
        self.encoders = nn.ModuleDict({
            name: NAIMEncoder(size, d_embed, n_heads, n_enc_layers, dropout)
            for name, size in MODALITY_SIZES.items()
        })

        total_tokens = sum(MODALITY_SIZES.values())   # 3+6+5+8+13 = 35
        self.shared_enc = SharedEncoder(
            total_tokens, d_embed, n_heads, n_shared_layers, n_classes, dropout
        )

    def forward(
        self,
        modality_data : Dict[str, torch.Tensor],   # name → (B, n_feat)
        feature_masks : Dict[str, torch.Tensor],   # name → (B, n_feat)
    ) -> torch.Tensor:
        """
        Returns logits (B, n_classes).
        """
        r_list      = []
        fmask_list  = []

        for name in MODALITY_NAMES:
            x = modality_data[name]          # (B, n_feat)
            m = feature_masks[name]          # (B, n_feat)
            r = self.encoders[name](x, m)    # (B, n_feat, d_embed)
            r_list.append(r)
            fmask_list.append(m)

        r_sh       = torch.cat(r_list,     dim=1)   # (B, total_tokens, d_embed)
        token_mask = torch.cat(fmask_list, dim=1)   # (B, total_tokens)

        return self.shared_enc(r_sh, token_mask)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  IT FUSION ORCHESTRATOR
#     Wraps the full pipeline: receives upstream outputs → builds JSON
# ═════════════════════════════════════════════════════════════════════════════

class ITFusion:
    """
    Stateful wrapper that:
      1. Calls ABCDE computation
      2. Builds modality tensors
      3. Runs MARIA for classification
      4. Generates unified Clinical JSON
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = MARIADermatology().to(self.device)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f'[ITFusion] Loaded weights from {model_path}')
        else:
            print('[ITFusion] No weights loaded — using random initialisation.')

    def run(
        self,
        # ── From ARCUNet (Stage 2)
        mask: np.ndarray,
        # ── From SLRC (Stage 3)
        roi_img: np.ndarray,
        # ── From HC (Stage 4)
        hc_predicted_class: str,
        hc_confidence: float,
        hc_shape_features: Optional[np.ndarray] = None,
        # ── From MedProc (Stage 5)
        evolution_score: Optional[float] = None,
        symptom_keywords: Optional[List[str]] = None,
        # ── From dataset_prep modality tensors
        tabular_modalities: Optional[Dict[str, np.ndarray]] = None,
        tabular_masks: Optional[Dict[str, np.ndarray]] = None,
        # ── Patient metadata for drug rules
        patient_history: Optional[dict] = None,
        # ── Physical calibration
        pixel_spacing_mm: float = 0.1,
    ) -> Dict:
        """
        Full IT Fusion pipeline.  Returns unified Clinical JSON dict.
        """

        # ── Step 1: Compute ABCDE ─────────────────────────────────────────
        abcde_result: ABCDEResult = compute_abcde(
            mask=mask,
            roi_img=roi_img,
            evolution_score=evolution_score,
            symptom_keywords=symptom_keywords,
            pixel_spacing_mm=pixel_spacing_mm,
        )

        # ── Step 2: ABCDE inference → clinical report ─────────────────────
        clinical_report: ClinicalReport = interpret_abcde(
            abcde_result, predicted_class=hc_predicted_class
        )

        # ── Step 3: ABCDE feature vector (new modality for MARIA) ─────────
        abcde_feats = abcde_to_feature_vector(abcde_result)
        abcde_arr   = np.array(list(abcde_feats.values()), dtype=np.float32)
        abcde_mask  = np.ones(len(abcde_arr), dtype=np.float32)   # always present

        # ── Step 4: Build complete modality tensors ────────────────────────
        if tabular_modalities is None:
            # Create minimal tensors if upstream data not provided
            tabular_modalities = {
                'demographics'    : np.zeros(3,  dtype=np.float32),
                'symptoms'        : np.zeros(6,  dtype=np.float32),
                'image_meta'      : np.zeros(5,  dtype=np.float32),
                'medical_history' : np.zeros(8,  dtype=np.float32),
            }
            tabular_masks = {k: np.zeros(v.shape, dtype=np.float32)
                             for k, v in tabular_modalities.items()}

        # Populate symptoms modality from MedProc keyword extraction
        # Symptom vector indices: [itch, grew, hurt, changed, bleed, elevation]
        SYMPTOM_KEYWORD_MAP = {
            0: {'itch', 'itching', 'pruritus'},
            1: {'grew', 'growth', 'larger', 'growing', 'spreading', 'enlarging',
                'increased size', 'increasing'},
            2: {'hurt', 'painful', 'pain', 'tender'},
            3: {'changed', 'changing', 'colour change', 'color change',
                'changed color', 'darkening'},
            4: {'bleed', 'bleeding', 'oozing', 'bleeding from'},
            5: {'elevation', 'elevated', 'raised', 'nodule'},
        }
        if symptom_keywords and tabular_masks.get('symptoms') is not None:
            sym_vec = tabular_modalities.get('symptoms',
                                             np.zeros(6, dtype=np.float32)).copy()
            kw_lower = {kw.lower().strip() for kw in symptom_keywords}
            matched_any = False
            for idx, kw_set in SYMPTOM_KEYWORD_MAP.items():
                if kw_lower & kw_set:  # set intersection
                    sym_vec[idx] = 1.0
                    matched_any = True
            if matched_any:
                tabular_modalities['symptoms'] = sym_vec.astype(np.float32)
                tabular_masks['symptoms'] = np.ones(6, dtype=np.float32)

        tabular_modalities['abcde'] = abcde_arr
        tabular_masks['abcde']      = abcde_mask

        # Add HC features into image_meta if available
        if hc_shape_features is not None:
            sf = hc_shape_features.flatten()[:5]          # take first 5
            sf = np.pad(sf, (0, max(0, 5 - len(sf))))     # pad to 5
            tabular_modalities['image_meta'] = sf.astype(np.float32)
            tabular_masks['image_meta']      = np.ones(5, dtype=np.float32)

        # ── Step 5: Run MARIA ──────────────────────────────────────────────
        with torch.no_grad():
            mod_data = {
                k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
                for k, v in tabular_modalities.items()
            }
            mod_masks = {
                k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
                for k, v in tabular_masks.items()
            }
            logits = self.model(mod_data, mod_masks)  # (1, n_classes)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]
            maria_class_idx = int(np.argmax(probs))
            maria_class     = CLASS_NAMES[maria_class_idx]
            maria_confidence= float(probs[maria_class_idx])

        # ── Step 6: Drug recommendations ──────────────────────────────────
        drug_rec = recommend_treatment(
            predicted_class=hc_predicted_class,
            risk_level=clinical_report.risk_level,
            patient_history=patient_history,
        )

        # ── Step 7: Build unified Clinical JSON ───────────────────────────
        clinical_json = {
            'classification': {
                'hc_class'       : hc_predicted_class,
                'hc_confidence'  : round(hc_confidence, 4),
                'maria_class'    : maria_class,
                'maria_confidence': round(maria_confidence, 4),
                'class_probs'    : {c: round(float(p), 4)
                                    for c, p in zip(CLASS_NAMES, probs)},
            },
            'ABCDE': {
                'A': round(abcde_result.A, 4),
                'B': round(abcde_result.B, 4),
                'C': round(abcde_result.C, 4),
                'D': round(abcde_result.D, 4),
                'E': round(abcde_result.E, 4),
            },
            'risk': {
                'score'      : round(clinical_report.risk_score, 4),
                'level'      : clinical_report.risk_level,
                'urgency'    : clinical_report.urgency,
                'n_criteria_flagged': clinical_report.flags.positive_criteria,
            },
            'clinical_summary' : clinical_report.summary,
            'evidence'         : clinical_report.evidence,
            'recommendations'  : clinical_report.recommendations,
            'differential_hints': clinical_report.differential_hints,
            'treatment': {
                'surgical_referral': drug_rec.surgical_referral,
                'oncology_referral': drug_rec.oncology_referral,
                'options': [
                    {
                        'name'  : opt.name,
                        'route' : opt.route,
                        'line'  : opt.line,
                        'note'  : opt.indication,
                        'ref'   : opt.reference,
                    }
                    for opt in drug_rec.options
                ],
                'general_notes': drug_rec.general_notes,
                'disclaimer'   : drug_rec.disclaimer,
            },
            'abcde_grades': {
                'A': clinical_report.flags.A_grade,
                'B': clinical_report.flags.B_grade,
                'C': clinical_report.flags.C_grade,
                'D': clinical_report.flags.D_grade,
                'E': clinical_report.flags.E_grade,
            },
        }

        return clinical_json


# ═════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def train_maria(
    model: MARIADermatology,
    train_loader,
    val_loader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu',
    save_path: str = 'maria_best.pt',
):
    """
    Standard training loop for MARIA with modality dropout regularisation.
    Expects DataLoader yielding:
      (modality_data_dict, feature_masks_dict, labels)
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    dev    = torch.device(device)
    model  = model.to(dev)
    opt    = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=n_epochs)
    crit   = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for mod_data, mod_masks, labels in train_loader:
            mod_data  = {k: v.to(dev) for k, v in mod_data.items()}
            mod_masks = {k: v.to(dev) for k, v in mod_masks.items()}
            labels    = labels.to(dev)

            opt.zero_grad()
            logits = model(mod_data, mod_masks)
            loss   = crit(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        sched.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for mod_data, mod_masks, labels in val_loader:
                mod_data  = {k: v.to(dev) for k, v in mod_data.items()}
                mod_masks = {k: v.to(dev) for k, v in mod_masks.items()}
                labels    = labels.to(dev)
                logits = model(mod_data, mod_masks)
                preds  = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += len(labels)

        val_acc = correct / (total + 1e-6)
        print(f'Epoch {epoch:3d}/{n_epochs} | '
              f'Train Loss: {train_loss/max(len(train_loader),1):.4f} | '
              f'Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'  ✓ Saved best model (val_acc={val_acc:.4f})')

    print(f'\nTraining complete. Best Val Acc: {best_val_acc:.4f}')
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 6.  DEMO
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import cv2

    print('=== IT Fusion Pipeline Demo ===\n')

    # Create synthetic ARCUNet mask output
    mask = np.zeros((224, 224), dtype=np.uint8)
    cv2.ellipse(mask, (112, 112), (65, 50), 20, 0, 360, 255, -1)
    cv2.circle(mask, (145, 85), 18, 255, -1)    # irregular bump

    # Create synthetic ROI
    roi = np.zeros((224, 224, 3), dtype=np.uint8)
    roi[50:175, 50:175] = [70, 35, 15]
    roi[85:130, 80:160] = [25, 55, 85]
    roi[105:145, 100:145] = [190, 190, 45]

    # HC stub outputs
    hc_class      = 'Melanoma'
    hc_confidence = 0.83
    hc_feats      = np.random.rand(5).astype(np.float32)

    # MedProc stub outputs
    evo_score = 0.72
    keywords  = ['grew larger', 'bleeding', 'changing colour']

    # PAD-UFES style tabular data (1 sample)
    tab_data = {
        'demographics'   : np.array([0.42, 1.0, 3.0], dtype=np.float32),
        'symptoms'        : np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        'image_meta'      : np.array([2.0, 0.0, 0.0, 0.65, 0.60], dtype=np.float32),
        'medical_history' : np.zeros(8, dtype=np.float32),
    }
    tab_masks = {
        'demographics'   : np.ones(3, dtype=np.float32),
        'symptoms'        : np.ones(6, dtype=np.float32),
        'image_meta'      : np.ones(5, dtype=np.float32),
        'medical_history' : np.zeros(8, dtype=np.float32),   # unknown history
    }

    patient_hist = {'asthma': False, 'autoimmune': False}

    # Run IT Fusion
    fusion = ITFusion()
    result = fusion.run(
        mask=mask,
        roi_img=roi,
        hc_predicted_class=hc_class,
        hc_confidence=hc_confidence,
        hc_shape_features=hc_feats,
        evolution_score=evo_score,
        symptom_keywords=keywords,
        tabular_modalities=tab_data,
        tabular_masks=tab_masks,
        patient_history=patient_hist,
    )

    print(json.dumps(result, indent=2))
