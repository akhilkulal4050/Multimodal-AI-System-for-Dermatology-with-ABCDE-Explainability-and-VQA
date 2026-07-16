"""
dermatology_bot.py — NutriDermAI Complete Inference + Conversation Server
==========================================================================

Pipeline per request:
    Raw image → ARCUNet (mask) → SLRC (bbox) → R-LLaVA (answer)

Imports ARCUNet and SLRC directly from your existing files.
No re-implementation of any existing code.

Usage:
    # Start server
    python dermatology_bot.py

    # Or with uvicorn for production
    uvicorn dermatology_bot:app --host 0.0.0.0 --port 8000

Requires:
    ARCUNet.py  — in the same directory (or ARCUNET_PY_PATH below)
    SLRC.py     — in the same directory (or SLRC_PY_PATH below)
    R-LLaVA checkpoint from Component 2

Models needed:
    1. ARCUNet checkpoint  → ARCUNET_CKPT
    2. R-LLaVA LoRA checkpoint → RLLAVA_CKPT  (from stage7_component2_rllava_finetune.ipynb)
    Base LLaVA model is downloaded automatically via unsloth.
"""

import os
import gc
import sys
import json
import uuid
import logging
import importlib.util
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# FastAPI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("DermBot")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS  — edit these to match your system
# ══════════════════════════════════════════════════════════════════════════════

_BASE = Path(__file__).parent


# ARCUNet.py and SLRC.py — same directory as this file by default'|
ARCUNET_PY_PATH = _BASE / "ARCUNet.py"
SLRC_PY_PATH    = _BASE / "SLRC.py"

# Model checkpoints
ARCUNET_CKPT    = Path("/home/vjti-comp/Desktop/Final Project Code/SLSf/arcunet_best_v3.pt")
# Checkpoint saved by nutriderm_rllava_train.py on DGX
# Copy from DGX: scp -r prasannam24-26@172.18.33.4:/home/prasannam24-26/rllava/checkpoints/rllava_stage2_best .
RLLAVA_CKPT     = Path("/home/vjti-comp/Desktop/Final Project Code/VQA/rllava/rllava/checkpoints/rllava_stage2_best")
# Base model is stored locally in the VQA folder — no HF download needed
BASE_MODEL_ID   = "/home/vjti-comp/Desktop/Final Project Code/VQA/llava-v1.6-mistral-7b-hf"

# Session storage
SESSIONS_DIR    = Path("/home/vjti-comp/Desktop/Final Project Code/VQA/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Inference settings
ARCUNET_THRESH  = 0.5       # use best_thresh from ARCUNet_Train2 val set if available
MAX_NEW_TOKENS  = 128
MAX_TURNS       = 10        # turns before context compression
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC IMPORT — ARCUNet.py and SLRC.py
# Imports them as modules from their actual file paths.
# No code is re-implemented here.
# ══════════════════════════════════════════════════════════════════════════════

def _import_from_path(name: str, path: Path):
    """Dynamically import a .py file as a module by file path."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place ARCUNet.py and SLRC.py in the same "
            f"directory as dermatology_bot.py, or update the path constants above."
        )
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

log.info("Importing ARCUNet and SLRC from source files...")
arcunet_module = _import_from_path("ARCUNet", ARCUNET_PY_PATH)
slrc_module    = _import_from_path("SLRC",    SLRC_PY_PATH)

# Pull the exact functions/classes we need — no re-implementation
ARCUNet          = arcunet_module.ARCUNet
load_model       = arcunet_module.load_model      # load_model(path, device, dropout_p)
predict_proba    = arcunet_module.predict_proba   # predict_proba(model, tensor, device) → np (H,W) float
slrc_from_logits = slrc_module.slrc_from_logits   # slrc_from_logits(img_np, logit_tensor, threshold, ...)

log.info("ARCUNet and SLRC imported successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# DERMATOLOGY RULES ENGINE — structured clinical RAG
# Replaces the simple CLINICAL_KB keyword lookup with a full rules engine that:
#   - Interprets ABCDE scores using clinical thresholds
#   - Maps risk_score to risk tier and recommended action
#   - Provides disease-specific treatment, differential diagnosis, follow-up
#   - Handles follow-up questions even when no disease keyword is in the question
# ══════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# DermatologyRulesEngine
# ════════════════════════════════════════════════════════════════════════════
# A structured clinical rules engine that converts ABCDE scores + risk_score
# + disease_label into a rich context string injected into every R-LLaVA prompt.
#
# Why this matters for R-LLaVA prediction quality:
#   R-LLaVA is a fine-tuned VQA model, not a clinical expert system.
#   Without explicit rules in the prompt, it cannot:
#     - Interpret ABCDE threshold crossings ("A=0.87 means HIGH asymmetry")
#     - Map risk scores to actions ("score 0.88 = urgent referral")
#     - Select the correct first-line treatment for a given diagnosis
#     - Explain WHY a lesion is concerning using ABCDE evidence
#   The rules engine computes these facts BEFORE the model runs and injects
#   them as grounding context, so the model is guided to correct answers.
#
# Architecture: Rule-Based Expert System (deterministic, no ML)
#   Input : disease_label, A, B, C, D, E, risk_score, question_intent
#   Output: structured context string (3-5 sentences, ~200 tokens)
# ════════════════════════════════════════════════════════════════════════════

# ── ABCDE Clinical Thresholds ─────────────────────────────────────────────
# Based on dermoscopy literature (Argenziano et al., Stolz et al.)
ABCDE_THRESHOLDS = {
    'A': {'low': 0.3,  'high': 0.6,
          'name': 'Asymmetry',
          'low_meaning':  'Symmetric lesion — low concern.',
          'mid_meaning':  'Mild asymmetry — warrants monitoring.',
          'high_meaning': 'Significant asymmetry — atypical, concerning.'},
    'B': {'low': 0.3,  'high': 0.6,
          'name': 'Border',
          'low_meaning':  'Well-defined, regular border — low concern.',
          'mid_meaning':  'Mildly irregular border — monitor for changes.',
          'high_meaning': 'Irregular, notched or indistinct border — atypical.'},
    'C': {'low': 0.25, 'high': 0.5,
          'name': 'Color',
          'low_meaning':  'Uniform color — low concern.',
          'mid_meaning':  'Mild color variation — monitor.',
          'high_meaning': 'Multiple colors or variegated pattern — atypical, concerning.'},
    'D': {'low': 0.3,  'high': 0.6,
          'name': 'Diameter/Dermoscopic structures',
          'low_meaning':  'Low structural complexity — benign-appearing.',
          'mid_meaning':  'Moderate structural features — monitor.',
          'high_meaning': 'Complex dermoscopic structures — atypical.'},
    'E': {'low': 0.3,  'high': 0.6,
          'name': 'Evolution',
          'low_meaning':  'Stable — no recent changes documented.',
          'mid_meaning':  'Some evolution noted — re-evaluate in 3 months.',
          'high_meaning': 'Significant evolution — change in size, shape or colour is a red flag.'},
}

# ── Risk Score → Tier Mapping ──────────────────────────────────────────────
RISK_TIERS = [
    (0.0,  0.30, 'LOW',      'Low risk. Routine monitoring recommended.'),
    (0.30, 0.55, 'MODERATE', 'Moderate risk. Dermatologist evaluation within 4-8 weeks.'),
    (0.55, 0.75, 'HIGH',     'High risk. Dermatologist evaluation within 2 weeks.'),
    (0.75, 1.01, 'URGENT',   'Urgent risk. Same-day or next-day dermatologist referral required.'),
]

# ── Disease → Clinical Profile ────────────────────────────────────────────
# Each entry contains: full name, malignancy status, first-line treatment,
# key differential features, follow-up protocol, red flag signs
DISEASE_PROFILES = {
    'MEL': {
        'name':          'Melanoma',
        'malignant':     True,
        'treatment':     ('Wide local excision with sentinel lymph node biopsy. '
                          'Surgical margins: 1cm for T1, 1-2cm for T2, 2cm for T3/T4. '
                          'Immunotherapy (pembrolizumab/nivolumab) for advanced stages.'),
        'differential':  ('Distinguish from dysplastic nevus by presence of all 5 ABCDE features. '
                          'Blue-black colour and ulceration are high-concern features. '
                          'Regression structures (grey-blue) on dermoscopy are red flags.'),
        'followup':      'Surgical excision within 2 weeks. Full-body skin examination. '
                         'Lymph node assessment. Annual full-body screening thereafter.',
        'red_flags':     ['rapid growth', 'ulceration', 'bleeding', 'satellite lesions',
                          'blue-black colour', 'regression'],
        'icd10':         'C43',
    },
    'BCC': {
        'name':          'Basal Cell Carcinoma',
        'malignant':     True,
        'treatment':     ('Surgical excision (first-line). Mohs micrographic surgery for high-risk '
                          'locations (face, ears). Topical imiquimod or 5-FU for superficial BCC. '
                          'Vismodegib for locally advanced or metastatic BCC.'),
        'differential':  ('Distinguish from SCC by pearly, translucent appearance and telangiectasia. '
                          'Nodular BCC most common (60%). Superficial BCC appears as pink flat patch. '
                          'Morphoeic/sclerosing BCC is difficult to detect clinically.'),
        'followup':      'Excision with clear margins (4-5mm). Annual skin checks for 5 years. '
                         'High recurrence risk on face/ears — consider Mohs surgery.',
        'red_flags':     ['ulceration', 'bleeding', 'pearly border', 'telangiectasia', 'rolled edges'],
        'icd10':         'C44.9',
    },
    'SCC': {
        'name':          'Squamous Cell Carcinoma',
        'malignant':     True,
        'treatment':     ('Surgical excision with 4-6mm margins (first-line). '
                          'Mohs surgery for high-risk locations. Radiation therapy if inoperable. '
                          'Cemiplimab for advanced/metastatic SCC.'),
        'differential':  ('Distinguish from BCC by firm, keratotic surface and lack of translucency. '
                          'Arises from actinic keratosis. Sun-exposed areas predominate (face, ears, hands). '
                          'Immunosuppressed patients have 100x higher SCC risk.'),
        'followup':      'Excision within 2 weeks. Lymph node assessment for high-risk cases. '
                         '3-monthly checks for 2 years, then annually.',
        'red_flags':     ['rapid growth', 'ulceration', 'lymphadenopathy', 'induration', 'pain'],
        'icd10':         'C44.92',
    },
    'ACK': {
        'name':          'Actinic Keratosis',
        'malignant':     False,
        'treatment':     ('Cryotherapy (first-line for isolated lesions). '
                          'Topical 5-fluorouracil (5-FU) cream for field treatment. '
                          'Topical imiquimod 5% for field treatment. '
                          'Photodynamic therapy (PDT) for extensive field cancerisation. '
                          'Diclofenac 3% gel for mild isolated lesions.'),
        'differential':  ('Precancerous; 5-10% progress to SCC over 10 years. '
                          'Rough, scaly patch on chronically sun-exposed skin. '
                          'Distinguish from SCC by lack of induration and slower growth. '
                          'Hyperkeratotic or hypertrophic AK has higher SCC progression risk.'),
        'followup':      '6-monthly dermatologist review. Sun protection counselling. '
                         'Treat all lesions to prevent SCC progression.',
        'red_flags':     ['induration', 'rapid thickening', 'ulceration', 'bleeding'],
        'icd10':         'L57.0',
    },
    'NEV': {
        'name':          'Melanocytic Nevus (benign mole)',
        'malignant':     False,
        'treatment':     ('No treatment required for stable, benign nevi. '
                          'Surgical excision if atypical features develop or patient concern. '
                          'Dermoscopic monitoring every 6-12 months for dysplastic nevi.'),
        'differential':  ('Benign: symmetric, well-defined border, uniform tan/brown colour. '
                          'Dysplastic nevus: asymmetry, irregular border, colour variation. '
                          'Clark/dysplastic nevus has higher melanoma risk if multiple (>50 nevi). '
                          'Blue nevus: deep blue-black, benign but resembles nodular melanoma.'),
        'followup':      'Annual full-body skin examination. '
                         'Document with serial dermoscopy. '
                         'Remove if ABCDE changes occur.',
        'red_flags':     ['rapid change', 'new ABCDE features', 'itching', 'bleeding', 'satellite lesions'],
        'icd10':         'D22',
    },
    'SEK': {
        'name':          'Seborrheic Keratosis',
        'malignant':     False,
        'treatment':     ('No treatment required — entirely benign. '
                          'Cryotherapy or curettage for cosmetic removal if desired. '
                          'Laser ablation as an alternative cosmetic option.'),
        'differential':  ('Benign: stuck-on, warty appearance. Uniform, cerebriform surface. '
                          'Horn cysts and milia-like cysts on dermoscopy are diagnostic. '
                          'Distinguish from melanoma: regular surface, no atypical vascularity. '
                          'Irritated SK may mimic SCC — dermoscopy essential.'),
        'followup':      'No follow-up required unless diagnosis uncertain. '
                         'Single check to confirm diagnosis if atypical features present.',
        'red_flags':     ['rapid change', 'bleeding', 'irregular vascular pattern'],
        'icd10':         'L82',
    },
}

# ── Question Intent Classifier ─────────────────────────────────────────────
# Maps question patterns to intent categories so rules can be intent-specific
INTENT_PATTERNS = {
    'diagnosis':      ['what condition','what skin','what disease','what is this','identify','diagnose'],
    'abcde':          ['abcde','asymmetr','border','colour','color','diameter','evolut','features'],
    'risk':           ['risk','serious','dangerous','concern','worr','urgent','how bad'],
    'treatment':      ['treatment','treat','cure','remove','surgery','medication','drug','cream'],
    'prognosis':      ['prognosis','outlook','survive','heal','recover','spread','metastas'],
    'followup':       ['follow','monitor','check','watch','next step','what should','recommend'],
    'differential':   ['differ','rule out','versus','vs','other possib','could it be'],
    'malignancy':     ['benign','malignant','cancer','malignancy','dangerous'],
    'cause':          ['cause','why','origin','risk factor','sun','uv','heredit','genetic'],
    'appearance':     ['look like','appear','describe','what does','visual','see','show'],
}

def classify_intent(question: str) -> list:
    """Return list of matched intent categories for the question."""
    q = question.lower()
    intents = []
    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in q for p in patterns):
            intents.append(intent)
    return intents if intents else ['general']

# ── ABCDE Interpreter ─────────────────────────────────────────────────────
def interpret_abcde(A=None, B=None, C=None, D=None, E=None) -> dict:
    """
    Apply clinical threshold rules to ABCDE scores.
    Returns dict with per-feature interpretation and overall summary.
    """
    findings = {}
    atypical_count = 0
    
    for feat, val in [('A',A),('B',B),('C',C),('D',D),('E',E)]:
        if val is None or float(val) < 0:
            findings[feat] = None
            continue
        v = float(val)
        thresh = ABCDE_THRESHOLDS[feat]
        if v >= thresh['high']:
            level = 'HIGH'
            text  = thresh['high_meaning']
            atypical_count += 1
        elif v >= thresh['low']:
            level = 'MODERATE'
            text  = thresh['mid_meaning']
        else:
            level = 'LOW'
            text  = thresh['low_meaning']
        findings[feat] = {'value': round(v,3), 'level': level, 'text': text,
                          'name': thresh['name']}
    
    return {'features': findings, 'atypical_count': atypical_count}

# ── Risk Tier Resolver ────────────────────────────────────────────────────
def get_risk_tier(risk_score: float) -> dict:
    """Map risk_score to clinical tier and action."""
    rs = float(risk_score) if risk_score is not None else 0.0
    for lo, hi, tier, action in RISK_TIERS:
        if lo <= rs < hi:
            return {'score': round(rs,3), 'tier': tier, 'action': action}
    return {'score': round(rs,3), 'tier': 'UNKNOWN', 'action': 'Clinical evaluation recommended.'}

# ════════════════════════════════════════════════════════════════════════════
# MAIN RULES ENGINE
# ════════════════════════════════════════════════════════════════════════════

class DermatologyRulesEngine:
    """
    Structured clinical rules engine for dermatology RAG.
    
    Flow:
      1. Receive question + session state (disease, ABCDE scores, risk_score)
      2. Classify question intent
      3. Apply intent-specific clinical rules
      4. Compose structured context string for R-LLaVA prompt injection
    
    The context string tells the model:
      - What the ABCDE scores MEAN clinically (not just the numbers)
      - What risk tier the score maps to and what action it implies
      - Which treatment is indicated for the disease
      - What differential diagnoses to consider
    """
    
    def __init__(self):
        self.disease_profiles = DISEASE_PROFILES
        self.abcde_thresholds = ABCDE_THRESHOLDS
        self.risk_tiers       = RISK_TIERS
    
    def get_context(
        self,
        question:      str,
        disease_label: str  = None,
        A: float = None, B: float = None, C: float = None,
        D: float = None, E: float = None,
        risk_score:    float = None,
        risk_level:    str   = None,
        clinical_summary: str = None,
        session_disease: str = None,
    ) -> str:
        """
        Main entry point. Returns clinical context string for prompt injection.
        
        All parameters are optional — the engine uses whatever is available.
        The more context provided, the richer and more specific the output.
        """
        # Resolve disease: use explicitly passed label, fall back to session disease
        disease = (disease_label or session_disease or '').upper()[:3]
        disease = disease if disease in self.disease_profiles else None
        
        # Classify what the question is asking about
        intents = classify_intent(question)
        
        # Interpret ABCDE if scores are available
        abcde = None
        has_abcde = any(v is not None and float(v if v is not None else -1) >= 0
                        for v in [A,B,C,D,E])
        if has_abcde:
            abcde = interpret_abcde(A, B, C, D, E)
        
        # Resolve risk tier
        risk_info = None
        if risk_score is not None:
            try: risk_info = get_risk_tier(float(risk_score))
            except Exception: pass
        
        # Build context based on intent
        parts = []
        
        # ── Disease identity context ──────────────────────────────────────
        if disease and ('diagnosis' in intents or 'appearance' in intents
                        or 'malignancy' in intents or 'general' in intents):
            prof = self.disease_profiles[disease]
            malign_str = "MALIGNANT" if prof['malignant'] else "BENIGN"
            parts.append(f"{prof['name']} [{malign_str}]: {prof['differential'][:120]}")
        
        # ── ABCDE interpretation ──────────────────────────────────────────
        if abcde and ('abcde' in intents or 'diagnosis' in intents
                      or 'risk' in intents or 'appearance' in intents):
            feat_lines = []
            for feat_key in ['A','B','C','D','E']:
                f = abcde['features'].get(feat_key)
                if f and f['level'] in ('HIGH','MODERATE'):
                    feat_lines.append(
                        f"{f['name']} {feat_key}={f['value']} [{f['level']}]: {f['text']}"
                    )
            if feat_lines:
                parts.append("ABCDE findings: " + " | ".join(feat_lines[:3]))
            
            n_atyp = abcde['atypical_count']
            if n_atyp >= 3:
                parts.append(f"{n_atyp}/5 ABCDE criteria are atypical — high concern for malignancy.")
            elif n_atyp >= 1:
                parts.append(f"{n_atyp}/5 ABCDE criteria are atypical — warrants evaluation.")
        
        # ── Risk tier and action ──────────────────────────────────────────
        if risk_info and ('risk' in intents or 'followup' in intents
                          or 'malignancy' in intents or 'diagnosis' in intents):
            parts.append(
                f"Risk assessment: score={risk_info['score']} "
                f"[{risk_info['tier']} RISK] — {risk_info['action']}"
            )
        
        # ── Treatment context ─────────────────────────────────────────────
        if disease and ('treatment' in intents or 'followup' in intents):
            prof = self.disease_profiles[disease]
            parts.append(f"Treatment for {prof['name']}: {prof['treatment'][:180]}")
        
        # ── Prognosis context ─────────────────────────────────────────────
        if disease and 'prognosis' in intents:
            prof = self.disease_profiles[disease]
            parts.append(f"Follow-up protocol: {prof['followup'][:150]}")
        
        # ── Red flags ─────────────────────────────────────────────────────
        if disease and ('risk' in intents or 'malignancy' in intents):
            prof = self.disease_profiles[disease]
            flags = prof['red_flags']
            parts.append(f"Red flags to watch for: {', '.join(flags[:4])}.")
        
        # ── Differential diagnosis ────────────────────────────────────────
        if disease and 'differential' in intents:
            prof = self.disease_profiles[disease]
            parts.append(f"Differential diagnosis: {prof['differential'][:200]}")
        
        # ── Fallback: if no intent matched but disease known ──────────────
        if not parts and disease:
            prof = self.disease_profiles[disease]
            malign_str = "malignant" if prof['malignant'] else "benign"
            parts.append(
                f"{prof['name']} is a {malign_str} condition. "
                f"{prof['differential'][:100]}"
            )
            if risk_info:
                parts.append(f"Risk: {risk_info['tier']} (score={risk_info['score']}). {risk_info['action']}")
        
        # Compose final context string (cap at ~250 tokens = ~200 chars)
        context = ' '.join(parts)
        return context[:500] if context else ''
    
    def get_full_assessment(
        self,
        disease_label: str = None,
        A=None, B=None, C=None, D=None, E=None,
        risk_score: float = None,
        clinical_summary: str = None,
    ) -> dict:
        """
        Returns a full structured assessment dict (used for session metadata).
        Called once when an image is first processed.
        """
        disease = (disease_label or '').upper()[:3]
        disease = disease if disease in self.disease_profiles else None
        
        abcde    = interpret_abcde(A,B,C,D,E) if any(v is not None for v in [A,B,C,D,E]) else {}
        risk     = get_risk_tier(float(risk_score)) if risk_score is not None else {}
        profile  = self.disease_profiles.get(disease, {}) if disease else {}
        
        return {
            'disease'       : disease,
            'disease_name'  : profile.get('name','Unknown'),
            'is_malignant'  : profile.get('malignant', None),
            'abcde'         : abcde,
            'risk'          : risk,
            'treatment'     : profile.get('treatment',''),
            'followup'      : profile.get('followup',''),
            'red_flags'     : profile.get('red_flags',[]),
            'icd10'         : profile.get('icd10',''),
        }


# Singleton instance — import and reuse everywhere
rules_engine = DermatologyRulesEngine()



def retrieve_rag_context(question: str, disease_label: str = None,
                         A=None, B=None, C=None, D=None, E=None,
                         risk_score=None, risk_level=None,
                         session_disease=None) -> str:
    """
    Wrapper around DermatologyRulesEngine.get_context() for backwards compatibility.
    Called in DermatologyBot.ask() before building the prompt.
    """
    return rules_engine.get_context(
        question=question, disease_label=disease_label,
        A=A, B=B, C=C, D=D, E=E,
        risk_score=risk_score, risk_level=risk_level,
        session_disease=session_disease,
    )

# ══════════════════════════════════════════════════════════════════════════════
# ARCUNet preprocessing transform
# Must match the resize/normalize pipeline used in ARCUNet_Train2.
# If your training notebook used different mean/std stats, update them here.
# ══════════════════════════════════════════════════════════════════════════════
_ARCUNET_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def remove_hair(image_np: np.ndarray) -> np.ndarray:
    """
    Black-hat morphological hair removal matching ARCUNet_Train2 preprocessing.

    Steps:
        1. Convert to grayscale
        2. Black-hat transform with large kernel (detects thin dark lines = hair)
        3. Threshold the black-hat result → hair mask
        4. Inpaint hair pixels using surrounding skin texture (Telea algorithm)

    Returns RGB uint8 image with hair inpainted.
    """
    gray    = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Dilate mask slightly so inpainting covers full hair width
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
    # Inpaint with Telea — fast, preserves lesion colour fidelity
    bgr     = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    inpaint = cv2.inpaint(bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)


def preprocess_for_arcunet(pil_image: Image.Image) -> torch.Tensor:
    """
    Prepare a PIL image for ARCUNet inference.

    Returns float32 tensor of shape (1, 3, 512, 512) on DEVICE.
    """
    img_np  = np.array(pil_image.convert("RGB"))
    img_np  = remove_hair(img_np)          # match ARCUNet_Train2 preprocessing
    img_pil = Image.fromarray(img_np)
    tensor  = _ARCUNET_TRANSFORM(img_pil).unsqueeze(0)  # (1,3,512,512)
    return tensor.to(DEVICE)


def get_bbox_and_roi(
    pil_image: Image.Image,
    arcunet_model: ARCUNet,
    threshold: float = ARCUNET_THRESH,
) -> tuple:
    """
    Full ARCUNet → SLRC pipeline for one image.

    Args:
        pil_image     : original PIL image (any resolution)
        arcunet_model : loaded ARCUNet in eval mode
        threshold     : sigmoid threshold (use best_thresh from training)

    Returns:
        roi_pil  : PIL Image  — 224×224 ROI crop (for reference / display)
        bbox     : (x, y, w, h) in original image coordinates
        roi_blended : PIL Image — original image with coloured bbox overlay
                      This is what gets passed to R-LLaVA for visual grounding.
    """
    img_np = np.array(pil_image.convert("RGB"))
    tensor = preprocess_for_arcunet(pil_image)

    # ARCUNet forward pass → raw logits (1,1,512,512)
    arcunet_model.eval()
    with torch.no_grad():
        logits = arcunet_model(tensor)       # eval mode: returns (B,1,H,W) directly

    # SLRC: sigmoid + threshold + bbox extraction
    # slrc_from_logits handles sigmoid→threshold→contour→bbox internally
    roi_np, (x, y, w, h) = slrc_from_logits(
        original_image  = img_np,
        logit_tensor    = logits,
        threshold       = threshold,
        output_size     = (224, 224),
        padding         = 10,
        min_area        = 100,
        use_convex_hull = True,
    )

    roi_pil = Image.fromarray(roi_np.astype(np.uint8))
    bbox    = (x, y, w, h)

    # Create RoI-blended image for R-LLaVA (alpha overlay of bbox on original)
    roi_blended = _blend_bbox(img_np, bbox, alpha=0.75)

    return roi_pil, bbox, roi_blended


def _blend_bbox(
    img_np: np.ndarray,
    bbox:   tuple,
    color:  tuple = (255, 0, 0),
    alpha:  float = 0.75,
    thickness: int = 3,
) -> Image.Image:
    """
    Draw a coloured bounding box overlay on the original image.
    Matches the alpha-blending used during R-LLaVA training in Component 2.

    alpha: weight on the overlay (border-highlighted) image.
    """
    x, y, w, h = [int(v) for v in bbox]
    overlay     = img_np.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
    blended = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)
    # Resize to 336×336 for R-LLaVA (matches training IMG_SIZE)
    blended = cv2.resize(blended, (336, 336), interpolation=cv2.INTER_AREA)
    return Image.fromarray(blended.astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY
# Persists session state: image, turn history, compressed older context.
# ══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    Per-session state for the dermatology bot.

    Stores:
        image          : current PIL image
        image_path     : path if loaded from disk
        bbox           : (x,y,w,h) from ARCUNet+SLRC
        blended_image  : 336×336 PIL with bbox overlay (fed to R-LLaVA)
        turns          : list of {role, content, ts} dicts
        context_sum    : compressed summary of turns beyond max_turns
    """

    def __init__(self, session_id: str = None, max_turns: int = MAX_TURNS):
        self.session_id   = session_id or str(uuid.uuid4())[:8]
        self.max_turns    = max_turns
        self.image        = None          # original PIL image
        self.image_path   = None
        self.bbox            = None
        self.blended_image   = None
        self.turns           = []
        self.context_sum     = ""
        self.session_disease = None    # disease code from ARCUNet or Stage 6
        self.abcde_scores    = {'A':None,'B':None,'C':None,'D':None,'E':None}
        self.risk_score      = None
        self.risk_level      = None
        self.session_assessment = {}   # full rules engine assessment
        self.created_at      = datetime.now().isoformat()
        self.updated_at      = self.created_at

    # ── Image registration ────────────────────────────────────────────────
    def set_image(self, pil_image: Image.Image, bbox: tuple, blended: Image.Image,
                  image_path: str = None):
        """Register a new image with its computed bbox and blended overlay."""
        self.image         = pil_image.convert("RGB")
        self.bbox          = bbox
        self.blended_image = blended
        self.image_path    = image_path
        # Reset conversation when a new image is registered
        self.turns         = []
        self.context_sum   = ""
        self.updated_at    = datetime.now().isoformat()
        log.info(f"[{self.session_id}] Image set, bbox={bbox}. Conversation reset.")

    # ── Turn management ───────────────────────────────────────────────────
    def add_turn(self, role: str, content: str):
        self.turns.append({
            "role"   : role,
            "content": content,
            "ts"     : datetime.now().isoformat(),
        })
        self.updated_at = datetime.now().isoformat()
        if len(self.turns) > self.max_turns * 2:
            self._compress()

    def _compress(self):
        """Compress oldest half of turns into a summary string."""
        half = len(self.turns) // 2
        old  = self.turns[:half]
        self.turns = self.turns[half:]
        parts = [self.context_sum] if self.context_sum else []
        for t in old:
            prefix = "Patient asked" if t["role"] == "user" else "Doctor noted"
            parts.append(f"{prefix}: {t['content'][:80]}")
        self.context_sum = " | ".join(parts[-6:])
        log.debug(f"[{self.session_id}] Compressed {half} turns.")

    # ── Prompt builder ────────────────────────────────────────────────────
    def build_prompt(self, question: str, bbox_str: str, rag_context: str = "") -> str:
        """
        Build the full R-LLaVA prompt with conversation history injected.

        Format matches training (Component 2):
          [INST] <image>
          Region of interest: [x,y,x+w,y+h]
          {compressed context}
          {recent conversation}
          Patient: {question} [/INST]
        """
        parts = [f"[INST] <image>\nRegion of interest: {bbox_str}"]

        if self.context_sum:
            parts.append(f"Previous context: {self.context_sum}")

        if self.turns:
            parts.append("Conversation so far:")
            for t in self.turns[-4:]:   # last 2 exchanges
                prefix = "Patient" if t["role"] == "user" else "Doctor"
                parts.append(f"{prefix}: {t['content'][:100]}")

        parts.append(f"Patient: {question} [/INST]")
        return "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self) -> Path:
        """Persist to JSON (image array not stored — only path reference)."""
        data = {
            "session_id" : self.session_id,
            "image_path" : self.image_path,
            "bbox"       : list(self.bbox) if self.bbox else None,
            "turns"      : self.turns,
            "context_sum": self.context_sum,
            "created_at" : self.created_at,
            "updated_at" : self.updated_at,
        }
        path = SESSIONS_DIR / f"{self.session_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def load(cls, session_id: str) -> "ConversationMemory":
        """Restore a session from disk (image reloaded from image_path if available)."""
        path = SESSIONS_DIR / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found.")
        with open(path) as f:
            data = json.load(f)
        mem              = cls(session_id=data["session_id"])
        mem.turns        = data.get("turns", [])
        mem.context_sum  = data.get("context_sum", "")
        mem.created_at   = data.get("created_at", "")
        mem.updated_at   = data.get("updated_at", "")
        mem.bbox         = tuple(data["bbox"]) if data.get("bbox") else None
        mem.image_path   = data.get("image_path")
        if mem.image_path and Path(mem.image_path).exists():
            mem.image = Image.open(mem.image_path).convert("RGB")
        return mem

    def get_history(self) -> list:
        return self.turns.copy()

    def __repr__(self):
        return (f"ConversationMemory(id={self.session_id}, "
                f"turns={len(self.turns)}, image={self.image is not None})")


# ══════════════════════════════════════════════════════════════════════════════
# DERMATOLOGY BOT
# Full pipeline: ARCUNet → SLRC → ConversationMemory → R-LLaVA
# ══════════════════════════════════════════════════════════════════════════════

class DermatologyBot:
    """
    NutriDermAI dermatology assistant.

    Integrates:
        ARCUNet  → skin lesion segmentation mask
        SLRC     → bounding box extraction from mask
        R-LLaVA  → visual question answering with conversation memory

    ARCUNet and SLRC are imported directly from your existing .py files.
    No code is duplicated.
    """

    def __init__(
        self,
        arcunet_model,
        rllava_model,
        processor,
        text_tok,
        arcunet_thresh: float = ARCUNET_THRESH,
        max_new_tokens:  int  = MAX_NEW_TOKENS,
    ):
        self.arcunet     = arcunet_model
        self.rllava      = rllava_model
        self.processor   = processor
        self.text_tok    = text_tok
        self.thresh      = arcunet_thresh
        self.max_tok     = max_new_tokens
        self.sessions: dict[str, ConversationMemory] = {}

    # ── Session management ────────────────────────────────────────────────
    def new_session(self) -> str:
        mem = ConversationMemory()
        self.sessions[mem.session_id] = mem
        log.info(f"New session: {mem.session_id}")
        return mem.session_id

    def get_or_load_session(self, session_id: str) -> ConversationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory.load(session_id)
        return self.sessions[session_id]

    # ── Image processing ──────────────────────────────────────────────────
    def process_image(self, session_id: str, pil_image: Image.Image,
                      image_path: str = None):
        """
        Run ARCUNet → SLRC on a new image for a session.
        Stores the bbox and blended overlay in ConversationMemory.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(session_id=session_id)
        mem = self.sessions[session_id]

        log.info(f"[{session_id}] Running ARCUNet + SLRC on image...")
        _, bbox, blended = get_bbox_and_roi(pil_image, self.arcunet, self.thresh)
        mem.set_image(pil_image, bbox, blended, image_path=image_path)
        log.info(f"[{session_id}] bbox={bbox}")
        return bbox

    # ── Question answering ────────────────────────────────────────────────
    def ask(self, session_id: str, question: str) -> str:
        """
        Answer a question using the stored image and conversation history.

        The blended (bbox-overlaid) image is passed to R-LLaVA, not the raw
        image, so the model sees the highlighted lesion region exactly as
        it was trained on in Component 2.
        """
        mem = self.sessions.get(session_id)
        if mem is None:
            return "Session not found. Please start a new session."
        if mem.blended_image is None:
            return "Please upload a dermatology image first."

        x, y, w, h = mem.bbox
        bbox_str    = f"[{x},{y},{x+w},{y+h}]"
        rag_ctx = retrieve_rag_context(
            question       = question,
            session_disease= mem.session_disease,
            A=mem.abcde_scores.get('A'), B=mem.abcde_scores.get('B'),
            C=mem.abcde_scores.get('C'), D=mem.abcde_scores.get('D'),
            E=mem.abcde_scores.get('E'),
            risk_score     = mem.risk_score,
            risk_level     = mem.risk_level,
        )
        prompt = mem.build_prompt(question, bbox_str, rag_ctx)

        try:
            answer = self._generate(prompt, mem.blended_image)
        except Exception as e:
            log.error(f"[{session_id}] Generation error: {e}")
            answer = "I encountered an error. Please try again."

        mem.add_turn("user",      question)
        mem.add_turn("assistant", answer)
        mem.save()
        return answer

    def _generate(self, prompt: str, image: Image.Image) -> str:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.rllava.generate(
                **inputs,
                max_new_tokens=self.max_tok,
                do_sample=False,
                pad_token_id=self.text_tok.pad_token_id,
            )
        answer = self.text_tok.decode(
            out[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        del inputs, out
        gc.collect()
        return answer or "Could not generate a response. Please rephrase."

    def get_history(self, session_id: str) -> list:
        mem = self.sessions.get(session_id)
        return mem.get_history() if mem else []

    def end_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].save()
            del self.sessions[session_id]
            log.info(f"Session {session_id} closed.")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_models():
    """
    Load ARCUNet + R-LLaVA and return an initialised DermatologyBot.

    Called once at server startup.
    """
    log.info("=" * 60)
    log.info("Loading models...")

    # 1. ARCUNet
    log.info(f"Loading ARCUNet from {ARCUNET_CKPT}")
    if not ARCUNET_CKPT.exists():
        raise FileNotFoundError(
            f"ARCUNet checkpoint not found: {ARCUNET_CKPT}\n"
            f"Update ARCUNET_CKPT in dermatology_bot.py."
        )
    arcunet = load_model(str(ARCUNET_CKPT), device=DEVICE, dropout_p=0.1)
    arcunet.eval()
    log.info(f"ARCUNet loaded on {DEVICE}.")

    # 2. R-LLaVA (fine-tuned LLaVA-v1.6-Mistral-7B + QLoRA adapter)
    log.info(f"Loading R-LLaVA from {RLLAVA_CKPT}")
    if not RLLAVA_CKPT.exists():
        raise FileNotFoundError(
            f"R-LLaVA checkpoint not found: {RLLAVA_CKPT}\n"
            f"Run stage7_component2_rllava_finetune.ipynb first."
        )

    gc.collect()
    torch.cuda.empty_cache()

    # Pure transformers + peft — matches how model was trained in nutriderm_rllava_train.py
    from transformers import (LlavaNextForConditionalGeneration,
                              LlavaNextProcessor, BitsAndBytesConfig)
    from peft import PeftModel

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # BASE_MODEL_ID already points to local folder — no HF download needed
    _model_src = BASE_MODEL_ID
    log.info(f"Loading base model from: {_model_src}")
    _base = LlavaNextForConditionalGeneration.from_pretrained(
        _model_src,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    _base.config.use_cache = False
    rllava = PeftModel.from_pretrained(_base, str(RLLAVA_CKPT))
    rllava.eval()

    processor = LlavaNextProcessor.from_pretrained(
        _model_src, local_files_only=True)
    text_tok  = processor.tokenizer
    if text_tok.pad_token is None:
        text_tok.pad_token    = text_tok.eos_token
        text_tok.pad_token_id = text_tok.eos_token_id

    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log.info(f"R-LLaVA loaded. VRAM used: {vram:.2f} GB")
    log.info("=" * 60)

    return DermatologyBot(arcunet, rllava, processor, text_tok)


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI SERVER
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "NutriDermAI — Dermatology Bot",
    description = "R-LLaVA dermatology assistant with ARCUNet lesion segmentation.",
    version     = "1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Global bot instance — loaded once at startup
bot: Optional[DermatologyBot] = None


@app.on_event("startup")
async def startup():
    global bot
    bot = load_all_models()
    log.info("Server ready.")


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status" : "ok",
        "model"  : "NutriDermAI R-LLaVA",
        "version": "1.0",
        "device" : DEVICE,
    }

@app.get("/health")
async def health():
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {"status": "ok", "vram_gb": round(vram, 2)}


# ── Session management ───────────────────────────────────────────────────────

@app.post("/session/new")
async def new_session():
    """Create a new conversation session."""
    sid = bot.new_session()
    return {
        "session_id": sid,
        "message"   : "Session created. POST an image to /session/{id}/image to begin.",
    }


@app.post("/session/{session_id}/image")
async def upload_image(
    session_id: str,
    file: UploadFile = File(...),
    # Optional: pass Stage 6 clinical data to enable rules engine from first question
    disease_label: Optional[str] = Form(None),
    risk_score:    Optional[float] = Form(None),
    risk_level:    Optional[str]   = Form(None),
    A: Optional[float] = Form(None), B: Optional[float] = Form(None),
    C: Optional[float] = Form(None), D: Optional[float] = Form(None),
    E: Optional[float] = Form(None),
):
    """
    Upload a dermatology image. ARCUNet+SLRC segments lesion and extracts bbox.
    Optionally pass Stage 6 clinical data (disease_label, ABCDE, risk_score)
    to enable the clinical rules engine from the very first question.
    """
    try:
        data  = await file.read()
        image = Image.open(BytesIO(data)).convert("RGB")
        bbox  = bot.process_image(session_id, image)
        x, y, w, h = bbox
        
        # Register clinical context if provided (from Stage 6 parquet lookup)
        mem = bot.sessions.get(session_id)
        if mem and any(v is not None for v in [disease_label,risk_score,A,B,C,D,E]):
            if disease_label:
                mem.session_disease = str(disease_label).upper()[:3]
            for feat, val in [('A',A),('B',B),('C',C),('D',D),('E',E)]:
                if val is not None:
                    try: mem.abcde_scores[feat] = float(val)
                    except: pass
            if risk_score is not None:
                try: mem.risk_score = float(risk_score)
                except: pass
            if risk_level: mem.risk_level = str(risk_level)
            mem.session_assessment = rules_engine.get_full_assessment(
                disease_label=mem.session_disease,
                A=mem.abcde_scores.get('A'), B=mem.abcde_scores.get('B'),
                C=mem.abcde_scores.get('C'), D=mem.abcde_scores.get('D'),
                E=mem.abcde_scores.get('E'), risk_score=mem.risk_score,
            )
            log.info(f"[{session_id}] Clinical context: disease={mem.session_disease} "
                     f"risk_tier={mem.session_assessment.get('risk',{}).get('tier','?')}")
        
        return {
            "session_id": session_id,
            "status"    : "image_processed",
            "bbox"      : {"x": x, "y": y, "w": w, "h": h},
            "disease"   : mem.session_disease if mem else None,
            "risk_tier" : mem.session_assessment.get('risk',{}).get('tier') if mem else None,
            "message"   : f"Lesion detected at ({x},{y}) size {w}x{h}. Ready for questions.",
        }
    except Exception as e:
        log.error(f"Image processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/ask")
async def ask_question(session_id: str, question: str = Form(...)):
    """
    Ask a question about the uploaded image.

    Conversation history is maintained automatically.
    The model answers in the context of all previous turns in this session.
    """
    try:
        answer      = bot.ask(session_id, question)
        history     = bot.get_history(session_id)
        turn_count  = len([t for t in history if t["role"] == "user"])
        return {
            "session_id": session_id,
            "question"  : question,
            "answer"    : answer,
            "turn_count": turn_count,
        }
    except Exception as e:
        log.error(f"QA error [{session_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_all_in_one(
    question  : str                    = Form(...),
    file      : Optional[UploadFile]   = File(None),
    session_id: Optional[str]          = Form(None),
):
    """
    All-in-one endpoint. Optionally include an image file.

    - If session_id is not provided, a new session is created automatically.
    - If a file is provided, it is processed by ARCUNet+SLRC before answering.
    - If no file and no existing session image, returns an error.

    This endpoint is designed for simple front-end integration where you
    want a single call to handle everything.
    """
    try:
        if session_id is None:
            session_id = bot.new_session()

        if file is not None:
            data  = await file.read()
            image = Image.open(BytesIO(data)).convert("RGB")
            bot.process_image(session_id, image)

        answer = bot.ask(session_id, question)
        return {"session_id": session_id, "answer": answer}
    except Exception as e:
        log.error(f"ask_all_in_one error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Conversation history ─────────────────────────────────────────────────────

@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Return the full conversation history for a session."""
    try:
        mem     = bot.get_or_load_session(session_id)
        history = mem.get_history()
        return {
            "session_id": session_id,
            "turns"     : history,
            "count"     : len(history),
            "bbox"      : list(mem.bbox) if mem.bbox else None,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a session. Saves conversation to disk."""
    bot.end_session(session_id)
    return {"session_id": session_id, "status": "closed"}


@app.get("/sessions")
async def list_sessions():
    """List all saved session IDs."""
    sessions = [p.stem for p in SESSIONS_DIR.glob("*.json")]
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/session/{session_id}/bbox")
async def get_bbox(session_id: str):
    """Return the bounding box computed for the session's image."""
    try:
        mem = bot.get_or_load_session(session_id)
        if mem.bbox is None:
            raise HTTPException(status_code=404, detail="No image in this session yet.")
        x, y, w, h = mem.bbox
        return {"session_id": session_id, "bbox": {"x": x, "y": y, "w": w, "h": h}}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dermatology_bot:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = False,       # set True during development
        log_level = "info",
    )