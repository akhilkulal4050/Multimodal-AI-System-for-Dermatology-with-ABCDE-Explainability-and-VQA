"""
uncertainty_detector.py
───────────────────────
Detects and quantifies uncertainty across all modalities and pipeline stages
in the IT Fusion system.

Three categories of uncertainty are tracked:
  1. Data Uncertainty   — missing modalities, low-quality inputs
  2. Model Uncertainty  — low confidence scores, close class probabilities
  3. Clinical Uncertainty — ABCDE borderline scores, conflicting signals

All flags are human-readable and suitable for inclusion in clinical JSON.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────

@dataclass
class UncertaintyReport:
    """Complete uncertainty assessment for one inference pass."""
    data_flags     : List[str]    # Missing / degraded input flags
    model_flags    : List[str]    # Low confidence / class collision flags
    clinical_flags : List[str]    # Borderline ABCDE / conflicting signal flags
    overall_level  : str          # 'Low' | 'Moderate' | 'High' | 'Critical'
    reliability_score : float     # 0 = unreliable, 1 = fully reliable
    should_refer   : bool         # Should this case be flagged for human review?
    summary        : str          # Plain-English one-liner


# ── THRESHOLDS ────────────────────────────────────────────────────────────────

# Confidence below which model is considered uncertain
CONF_THRESHOLD_LOW  = 0.50
CONF_THRESHOLD_MOD  = 0.70

# ABCDE borderline band (flags if score is between these values)
BORDERLINE_LOW  = 0.35
BORDERLINE_HIGH = 0.65

# Probability margin between top-2 classes to flag close call
MARGIN_THRESHOLD = 0.10


# ── MAIN INTERFACE ────────────────────────────────────────────────────────────

def detect_uncertainty(
    # ── Modality availability ──────────────────────────────────────────────────
    has_image       : bool = True,
    has_text        : bool = True,
    has_demographics: bool = True,
    has_history     : bool = True,
    has_abcde       : bool = True,

    # ── Image quality signals ──────────────────────────────────────────────────
    image_quality   : Optional[float] = None,   # 0=poor, 1=excellent
    mask_coverage   : Optional[float] = None,   # fraction of image that is lesion
    mask_area_px    : Optional[int]   = None,   # lesion area in pixels

    # ── Model confidence signals ───────────────────────────────────────────────
    hc_confidence   : Optional[float] = None,
    maria_confidence: Optional[float] = None,
    class_probs     : Optional[Dict[str, float]] = None,

    # ── ABCDE signals ─────────────────────────────────────────────────────────
    abcde_scores    : Optional[Dict[str, float]] = None,
    abcde_grades    : Optional[Dict[str, str]]   = None,
    risk_score      : Optional[float] = None,
    risk_level      : Optional[str]   = None,

    # ── Clinical conflict detection ────────────────────────────────────────────
    hc_predicted_class   : Optional[str] = None,
    maria_predicted_class: Optional[str] = None,

) -> UncertaintyReport:
    """
    Full uncertainty assessment for one IT Fusion inference.

    Returns
    -------
    UncertaintyReport — structured with flags, level, and human summary.
    """

    data_flags     = []
    model_flags    = []
    clinical_flags = []

    # ── 1. DATA UNCERTAINTY ───────────────────────────────────────────────────

    if not has_image:
        data_flags.append('missing_image: no skin lesion image provided')
    if not has_text:
        data_flags.append('missing_text: no clinical text / caption available')
    if not has_demographics:
        data_flags.append('missing_demographics: age/gender unknown')
    if not has_history:
        data_flags.append('missing_history: patient medical history not provided')
    if not has_abcde:
        data_flags.append('missing_abcde: ABCDE scores could not be computed '
                          '(mask may be invalid or empty)')

    if image_quality is not None and image_quality < 0.4:
        data_flags.append(f'low_image_quality: quality score {image_quality:.2f} '
                          f'— blurry, over-exposed, or partial lesion')

    if mask_coverage is not None and mask_coverage < 0.02:
        data_flags.append(f'small_lesion_region: mask covers only '
                          f'{mask_coverage:.1%} of image — colour analysis may be unreliable')

    if mask_area_px is not None and mask_area_px < 200:
        data_flags.append(f'tiny_lesion_mask: only {mask_area_px} lesion pixels detected '
                          f'— ABCDE computation is approximate')

    # ── 2. MODEL UNCERTAINTY ──────────────────────────────────────────────────

    if hc_confidence is not None:
        if hc_confidence < CONF_THRESHOLD_LOW:
            model_flags.append(f'low_hc_confidence: HC model confidence is '
                                f'{hc_confidence:.1%} — prediction may be unreliable')
        elif hc_confidence < CONF_THRESHOLD_MOD:
            model_flags.append(f'moderate_hc_confidence: HC model at {hc_confidence:.1%} '
                                f'— consider secondary review')

    if maria_confidence is not None:
        if maria_confidence < CONF_THRESHOLD_LOW:
            model_flags.append(f'low_maria_confidence: MARIA fusion confidence is '
                                f'{maria_confidence:.1%}')
        elif maria_confidence < CONF_THRESHOLD_MOD:
            model_flags.append(f'moderate_maria_confidence: MARIA at {maria_confidence:.1%}')

    if class_probs is not None and len(class_probs) >= 2:
        sorted_probs = sorted(class_probs.values(), reverse=True)
        top1, top2 = sorted_probs[0], sorted_probs[1]
        margin = top1 - top2
        if margin < MARGIN_THRESHOLD:
            top2_class = [k for k, v in class_probs.items() if abs(v - top2) < 1e-6]
            top1_class = [k for k, v in class_probs.items() if abs(v - top1) < 1e-6]
            model_flags.append(
                f'close_class_decision: margin between top classes '
                f'({top1_class[0]} {top1:.1%} vs {top2_class[0] if top2_class else "?"} {top2:.1%}) '
                f'is only {margin:.1%} — borderline classification'
            )

    # ── 3. CLINICAL UNCERTAINTY ───────────────────────────────────────────────

    if abcde_scores:
        borderline_criteria = []
        for crit, val in abcde_scores.items():
            if BORDERLINE_LOW <= val <= BORDERLINE_HIGH:
                borderline_criteria.append(f'{crit}={val:.2f}')
        if borderline_criteria:
            clinical_flags.append(
                f'borderline_abcde: criteria in ambiguous range '
                f'({", ".join(borderline_criteria)}) — additional imaging advised'
            )

    if abcde_grades and risk_score is not None:
        # E is severe but ABС are all normal — isolated evolution with clean appearance
        if (abcde_grades.get('E') in ('Moderate', 'Severe') and
                all(abcde_grades.get(c, 'Normal') == 'Normal' for c in 'ABC')):
            clinical_flags.append(
                'evolution_without_morphology: significant evolution signal but '
                'morphological criteria (A/B/C) appear normal — may indicate '
                'very early lesion change or documentation artefact'
            )

        # High colour but low asymmetry — could be pigmented BCC or benign dysplastic nevus
        if (abcde_grades.get('C') in ('Moderate', 'Severe') and
                abcde_grades.get('A') == 'Normal'):
            clinical_flags.append(
                'colour_without_asymmetry: multi-colour pattern with symmetric shape '
                '— differential includes BCC and dysplastic nevus'
            )

    # Model disagreement between HC and MARIA
    if hc_predicted_class and maria_predicted_class:
        hc_norm  = hc_predicted_class.lower().strip()
        mar_norm = maria_predicted_class.lower().strip()
        if hc_norm != mar_norm:
            # Check if they're in the same clinical group (e.g. MEL vs Melanoma)
            conflict = _check_model_conflict(hc_norm, mar_norm)
            if conflict:
                clinical_flags.append(
                    f'model_disagreement: HC predicts "{hc_predicted_class}" '
                    f'but MARIA fusion predicts "{maria_predicted_class}" '
                    f'— human review required'
                )

    if risk_score is not None and risk_level == 'High' and not (hc_confidence or 0) > 0.8:
        clinical_flags.append(
            'high_risk_low_confidence: risk score is elevated but model confidence '
            'is below 80% — clinical assessment should not rely solely on this output'
        )

    # ── 4. COMPUTE OVERALL LEVEL ──────────────────────────────────────────────

    n_flags = len(data_flags) + len(model_flags) + len(clinical_flags)
    has_critical = not has_image or (
        hc_confidence is not None and hc_confidence < 0.35
    )

    if has_critical or n_flags >= 5:
        overall = 'Critical'
        should_refer = True
    elif n_flags >= 3 or (data_flags and model_flags):
        overall = 'High'
        should_refer = True
    elif n_flags >= 1:
        overall = 'Moderate'
        should_refer = (risk_level in ('High', 'Moderate')) or bool(clinical_flags)
    else:
        overall = 'Low'
        should_refer = False

    # ── 5. RELIABILITY SCORE ──────────────────────────────────────────────────

    reliability = _compute_reliability(
        has_image, has_text, hc_confidence, maria_confidence,
        n_flags, abcde_scores
    )

    # ── 6. SUMMARY ────────────────────────────────────────────────────────────

    summary = _build_summary(overall, n_flags, data_flags, model_flags,
                             clinical_flags, reliability)

    return UncertaintyReport(
        data_flags     = data_flags,
        model_flags    = model_flags,
        clinical_flags = clinical_flags,
        overall_level  = overall,
        reliability_score = reliability,
        should_refer   = should_refer,
        summary        = summary,
    )


# ── RELIABILITY SCORER ────────────────────────────────────────────────────────

def _compute_reliability(
    has_image, has_text, hc_conf, maria_conf, n_flags, abcde_scores
) -> float:
    """Heuristic reliability [0,1]. 1 = fully reliable, 0 = unreliable."""
    score = 1.0

    if not has_image:
        score -= 0.40
    if not has_text:
        score -= 0.15

    if hc_conf is not None:
        score -= max(0, (CONF_THRESHOLD_MOD - hc_conf) * 1.5)
    if maria_conf is not None:
        score -= max(0, (CONF_THRESHOLD_MOD - maria_conf) * 0.8)

    # Each flag reduces reliability
    score -= n_flags * 0.05

    # Borderline ABCDE reduces reliability
    if abcde_scores:
        n_borderline = sum(
            1 for v in abcde_scores.values()
            if BORDERLINE_LOW <= v <= BORDERLINE_HIGH
        )
        score -= n_borderline * 0.04

    return max(0.0, min(1.0, round(score, 3)))


# ── MODEL CONFLICT CHECKER ────────────────────────────────────────────────────

# Maps similar class names to canonical groups for conflict detection
_CLASS_GROUPS = {
    'melanoma': 'melanoma', 'mel': 'melanoma',
    'bcc': 'bcc', 'basal cell': 'bcc',
    'scc': 'scc', 'squamous': 'scc',
    'akiec': 'ak', 'actinic': 'ak',
    'df': 'df', 'dermatofibroma': 'df',
    'vasc': 'vasc', 'vascular': 'vasc',
    'nev': 'nevus', 'nevus': 'nevus', 'naevus': 'nevus', 'mole': 'nevus',
}


def _check_model_conflict(hc: str, maria: str) -> bool:
    """Return True if the two predictions are in genuinely different clinical groups."""
    hc_g = next((v for k, v in _CLASS_GROUPS.items() if k in hc), hc)
    mr_g = next((v for k, v in _CLASS_GROUPS.items() if k in maria), maria)
    return hc_g != mr_g


# ── SUMMARY BUILDER ───────────────────────────────────────────────────────────

def _build_summary(
    level: str, n_flags: int,
    data_flags, model_flags, clinical_flags,
    reliability: float
) -> str:
    if level == 'Critical':
        return (
            f'Critical uncertainty ({n_flags} flags, reliability {reliability:.0%}). '
            'Major data or model issues detected — this output must NOT be used '
            'for clinical decisions without human expert review.'
        )
    elif level == 'High':
        return (
            f'High uncertainty ({n_flags} flags, reliability {reliability:.0%}). '
            'Multiple uncertainty sources detected. Dermatologist review is required.'
        )
    elif level == 'Moderate':
        return (
            f'Moderate uncertainty ({n_flags} flag{"s" if n_flags != 1 else ""}, '
            f'reliability {reliability:.0%}). '
            'Some signals require verification. Clinical follow-up is advised.'
        )
    else:
        return (
            f'Low uncertainty (reliability {reliability:.0%}). '
            'All modalities present and model confidence is adequate. '
            'Output is suitable for clinical decision support.'
        )


# ── CONVENIENCE: QUICK MODALITY CHECK ────────────────────────────────────────

def quick_modality_check(
    has_image: bool,
    has_text: bool,
    has_abcde: bool = True,
) -> List[str]:
    """
    Fast check returning a list of missing-modality flag strings.
    Used by legacy code or minimal pipelines.
    Backwards-compatible with the original 2-argument signature.
    """
    flags = []
    if not has_image:
        flags.append('missing_image')
    if not has_text:
        flags.append('missing_text')
    if not has_abcde:
        flags.append('missing_abcde')
    return flags


# ── DEMO ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json

    report = detect_uncertainty(
        has_image       = True,
        has_text        = True,
        has_demographics= False,
        has_history     = False,
        has_abcde       = True,
        image_quality   = 0.78,
        mask_coverage   = 0.12,
        mask_area_px    = 2800,
        hc_confidence   = 0.61,
        maria_confidence= 0.58,
        class_probs     = {'MEL': 0.58, 'BCC': 0.51, 'SCC': 0.03, 'AKIEC': 0.02,
                           'DF': 0.01, 'VASC': 0.01, 'NEV': 0.01},
        abcde_scores    = {'A': 0.48, 'B': 0.42, 'C': 0.61, 'D': 0.38, 'E': 0.77},
        abcde_grades    = {'A': 'Mild', 'B': 'Mild', 'C': 'Moderate', 'D': 'Mild', 'E': 'Severe'},
        risk_score      = 0.54,
        risk_level      = 'Moderate',
        hc_predicted_class   = 'Melanoma',
        maria_predicted_class= 'BCC',
    )

    print(f'Overall Uncertainty  : {report.overall_level}')
    print(f'Reliability Score    : {report.reliability_score:.1%}')
    print(f'Should Refer         : {report.should_refer}')
    print(f'\nData Flags:')
    for f in report.data_flags:
        print(f'  • {f}')
    print(f'Model Flags:')
    for f in report.model_flags:
        print(f'  • {f}')
    print(f'Clinical Flags:')
    for f in report.clinical_flags:
        print(f'  • {f}')
    print(f'\nSummary: {report.summary}')
