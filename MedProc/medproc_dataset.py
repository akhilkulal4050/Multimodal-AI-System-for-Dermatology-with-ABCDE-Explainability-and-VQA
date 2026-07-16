"""
MedProc Dataset Module
Handles data loading, preprocessing, cleaning, and utility functions for medical text processing.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path('/data/Stagewise Dataset/MedProc')
OUTPUT_DIR = Path('/data/Stagewise Dataset/MedProc/final5')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TEXT_LENGTH = 3000


# ─────────────────────────────────────────────────────────────────────────────
# ICD-9 Whitelist — Skin-related + Common diseases
# ─────────────────────────────────────────────────────────────────────────────

SKIN_ICD_PREFIXES = {
    # Malignant neoplasms of skin
    '172',  # Melanoma of skin
    '173',  # Other malignant neoplasm of skin
    '216',  # Benign neoplasm of skin
    '232',  # Carcinoma in situ of skin
    '238',  # Uncertain neoplasm of skin (238.2)
    '239',  # Unspecified neoplasm of skin (239.2)
    # Inflammatory / allergic skin conditions
    '690', '691', '692', '693', '694', '695', '696', '697', '698',
    '700', '701', '702', '703', '704', '705', '706', '707', '708', '709',
    # Infections of skin
    '680', '681', '682', '683', '684', '685', '686',
    # Viral skin conditions
    '054', '053', '078',
    # Burns
    '940', '941', '942', '943', '944', '945', '946',
}

COMMON_ICD_PREFIXES = {
    # Cardiovascular
    '401', '402', '403', '404', '405', '410', '411', '412', '413', '414',
    '427', '428', '430', '431', '432', '433', '434', '435', '436',
    # Respiratory
    '480', '481', '482', '483', '484', '485', '486', '491', '492', '493', '496', '518',
    # Endocrine / Metabolic
    '250', '244', '245', '272', '278',
    # Infectious
    '038', '041', '599', '110', '111', '112',
    # Renal
    '580', '581', '582', '583', '584', '585', '586',
    # Gastrointestinal
    '531', '532', '533', '534', '540', '541', '542', '550', '551', '552', '553',
    '570', '571', '572', '573', '574', '575', '576',
    # Blood
    '280', '281', '282', '283', '284', '285', '286', '287',
    # Neurological
    '345', '332', '340', '346',
    # Mental health
    '290', '291', '292', '293', '294', '295', '296', '300', '301', '303', '304', '305',
    # Musculoskeletal
    '710', '711', '712', '713', '714', '715', '716', '720', '721', '722', '723', '724',
    # Cancer (general)
    '140', '141', '142', '143', '144', '145', '150', '151', '152', '153', '154',
    '160', '161', '162', '174', '175', '179', '180', '182', '185', '186', '187', '188', '189',
    '200', '201', '202', '203', '204', '205',
}

ICD_WHITELIST = SKIN_ICD_PREFIXES | COMMON_ICD_PREFIXES


# ─────────────────────────────────────────────────────────────────────────────
# Text Preprocessing Functions
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_PATTERNS = [
    r'(assessment\s*(?:and|&)?\s*plan)[:\s]*',
    r'(impression\s*(?:and|&)?\s*plan)[:\s]*',
    r'(hospital\s*course)[:\s]*',
    r'(discharge\s*diagnosis)[:\s]*',
]
_SECTION_RE = re.compile('|'.join(_SECTION_PATTERNS), re.IGNORECASE)


def clean_text(text: str) -> str:
    """Remove PHI tags, normalize whitespace."""
    if text is None:
        return ''
    text = re.sub(r'\[\*\*.*?\*\*\]', '', str(text))  # Remove PHI tags
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def smart_truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """
    Prefer the Assessment/Plan section when truncating long notes.
    Falls back to the last `max_len` characters if section not found.
    """
    if len(text) <= max_len:
        return text
    match = _SECTION_RE.search(text)
    if match:
        section_text = text[match.start():]
        if len(section_text) <= max_len:
            return section_text
        return section_text[:max_len]
    return text[-max_len:]


# ─────────────────────────────────────────────────────────────────────────────
# ICD Filtering Functions
# ─────────────────────────────────────────────────────────────────────────────

def icd_is_relevant(icd_code) -> bool:
    """Return True if the ICD-9 code starts with any whitelisted prefix."""
    if pd.isna(icd_code) or icd_code is None:
        return False
    code = str(icd_code).strip()
    return any(code.startswith(prefix) for prefix in ICD_WHITELIST)


def filter_icd_column(df, icd_col: str, group_keys: List[str]):
    """Keep only rows where ICD code is in whitelist."""
    mask = df[icd_col].map(lambda c: icd_is_relevant(c), meta=(icd_col, 'bool'))
    return df[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Symptom/Evolution Keywords
# ─────────────────────────────────────────────────────────────────────────────

# ── EVOLUTION_KEYWORDS: skin-specific signals only ───────────────────────────
# Used for has_symptom training label — must be tightly scoped to skin evolution
# so the symptom_head learns "changing skin lesion" not "any clinical note"
EVOLUTION_KEYWORDS = [
    # Size / spread changes (unambiguous evolution signals)
    'growing', 'grown', 'enlarging', 'enlarged', 'spreading', 'spread',
    'increasing in size', 'getting bigger', 'getting larger',
    # Colour changes
    'color change', 'colour change', 'changing color', 'darkening', 'lightening',
    'new pigmentation', 'multicoloured', 'variegated',
    # Surface changes
    'bleeding', 'bleeding from', 'oozing', 'crusting', 'ulcerating', 'ulcerated',
    'scabbing', 'weeping',
    # Sensory changes
    'itching', 'pruritus', 'burning sensation', 'painful to touch',
    # Morphological change
    'new lesion', 'appeared recently', 'recently changed', 'changed in',
    'noticed change', 'change in shape', 'change in border',
    # Temporal context WITH skin reference
    'lesion for', 'spot for', 'mole for', 'growth for',
    'developed over', 'appeared over', 'present for',
    'rapid change', 'sudden change', 'gradual change',
]

# ── SYMPTOM_KEYWORDS: full list used for RUNTIME keyword extraction ────────────
# This broader list is used at inference to extract keywords that are
# passed to compute_evolution() as symptom_keywords for boost scoring.
# NOT used to create has_symptom training labels (that uses EVOLUTION_KEYWORDS).
SYMPTOM_KEYWORDS = [
    # Evolution signals (same as EVOLUTION_KEYWORDS above)
    'growing', 'enlarging', 'spreading', 'increasing', 'worsening',
    'bleeding', 'bleeding from', 'oozing', 'crusting', 'ulcerating',
    'itching', 'pruritus', 'burning', 'painful', 'tender',
    'color change', 'darkening', 'lighter', 'new lesion',
    'rapid change', 'sudden change', 'gradual change',
    'developed over', 'appeared over', 'recently changed',
    # Temporal WITH context (lesion/spot/mole explicitly)
    'lesion for weeks', 'lesion for months', 'lesion for years',
    'spot for weeks', 'mole for months', 'growth for',
    'sudden onset', 'gradual onset',
    # General symptoms (for general clinical context extraction)
    'fever', 'fatigue', 'nausea', 'vomiting', 'dyspnea', 'shortness of breath',
    'chest pain', 'palpitations', 'syncope', 'edema', 'swelling',
    'diarrhea', 'constipation', 'abdominal pain', 'jaundice',
    'headache', 'dizziness', 'weakness', 'confusion', 'seizure',
    'polyuria', 'polydipsia', 'weight loss', 'weight gain', 'anorexia',
    'hemoptysis', 'hematuria', 'dysuria', 'cough', 'sputum',
    # Skin-specific morphology
    'erythema', 'induration', 'vesicle', 'papule', 'plaque', 'nodule',
    'macule', 'pustule', 'bulla', 'scale', 'desquamation',
    'hyper-pigmentation', 'hypo-pigmentation', 'telangiectasia',
]

EVOLUTION_SET = set(EVOLUTION_KEYWORDS)
SYMPTOM_SET   = set(SYMPTOM_KEYWORDS)


def extract_symptoms_from_text(text: str) -> List[str]:
    """
    Rule-based symptom extraction for runtime keyword list.
    Used at inference — returns matched keywords from full SYMPTOM_KEYWORDS.
    These are passed to compute_evolution() as symptom_keywords for boost scoring.
    """
    text_lower = text.lower()
    found = [kw for kw in SYMPTOM_KEYWORDS if kw in text_lower]
    return found


def extract_evolution_signals(text: str) -> List[str]:
    """
    Skin-specific evolution signal extraction for has_symptom TRAINING LABEL.
    Uses tighter EVOLUTION_KEYWORDS (no bare time words, no general symptoms).
    This ensures the symptom_head learns "skin lesion evolution" not "any clinical note".

    IMPORTANT: Use this function (not extract_symptoms_from_text) to create
    the has_symptom label in medproc_train.ipynb Cell 4.
    """
    text_lower = text.lower()
    found = [kw for kw in EVOLUTION_KEYWORDS if kw in text_lower]
    return found


# ─────────────────────────────────────────────────────────────────────────────
# I/O Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_with_progress(ddf, output_path: str, label: str = 'Dataset'):
    """Save Dask DataFrame with progress bar."""
    print(f'[{label}] Writing Parquet...')
    with ProgressBar():
        ddf.to_parquet(str(output_path), write_index=False)
    print(f'[{label}] Done. Saved to {output_path}')


def load_dataset_parquet(path: str):
    """Load dataset from Parquet format."""
    return dd.read_parquet(str(path))


if __name__ == '__main__':
    print('MedProc Dataset Module Loaded')
    print(f'  ICD Whitelist: {len(ICD_WHITELIST)} prefixes')
    print(f'    - Skin: {len(SKIN_ICD_PREFIXES)}')
    print(f'    - Common: {len(COMMON_ICD_PREFIXES)}')
    print(f'  Symptom Keywords: {len(SYMPTOM_KEYWORDS)}')
