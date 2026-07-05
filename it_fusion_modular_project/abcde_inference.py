"""
abcde_inference.py
──────────────────
Converts raw ABCDE scores into actionable clinical reasoning.

This is a rule-based expert system that mirrors the clinical ABCDE
checklist used by dermatologists.  It is deterministic and explainable —
every flag traces back to a specific score and threshold.

Flow:
  ABCDEResult → ClinicalFlags → ClinicalReport → JSON for VQA / RAG

Also provides:
  - Feature vector builder (for MARIA fusion input)
  - Urgency classifier
"""

import json
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from abcde_computation import ABCDEResult


# ── CLINICAL THRESHOLDS ───────────────────────────────────────────────────────
# Based on standard dermoscopy teaching (AAD/EAD guidelines):

THRESHOLDS = {
    # (mild, moderate, severe) cut-offs for each criterion
    'A': (0.30, 0.55, 0.75),
    'B': (0.30, 0.55, 0.75),
    'C': (0.25, 0.50, 0.70),
    'D': (0.35, 0.55, 0.80),
    'E': (0.20, 0.45, 0.70),
}


def _grade(value: float, thresholds: tuple) -> str:
    mild, moderate, severe = thresholds
    if value >= severe:   return 'Severe'
    if value >= moderate: return 'Moderate'
    if value >= mild:     return 'Mild'
    return 'Normal'


# ── CLINICAL FLAG DATA CLASS ──────────────────────────────────────────────────

@dataclass
class ClinicalFlags:
    """Per-criterion graded assessment."""
    A_grade: str
    B_grade: str
    C_grade: str
    D_grade: str
    E_grade: str
    positive_criteria: int   # how many of A–E are ≥ Mild
    alarm_criteria: int      # how many of A–E are ≥ Moderate
    urgent_criteria: int     # how many of A–E are Severe


@dataclass
class ClinicalReport:
    """Full structured clinical output for VQA / RAG / display."""
    abcde_scores: dict
    flags: ClinicalFlags
    risk_score: float
    risk_level: str
    urgency: str             # 'Routine' | 'Soon (4-6 wks)' | 'Urgent (≤2 wks)' | 'Emergency'
    summary: str             # 1-2 sentence plain-English summary
    recommendations: List[str]
    differential_hints: List[str]
    evidence: List[str]
    raw_metrics: dict

    def to_dict(self):
        d = asdict(self)
        d['flags'] = asdict(self.flags)
        return d

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)


# ── URGENCY CLASSIFIER ────────────────────────────────────────────────────────

def classify_urgency(flags: ClinicalFlags, risk_score: float) -> str:
    """
    Rule-based urgency tier.

    Emergency : ≥3 severe criteria OR risk > 0.85
    Urgent    : ≥2 alarm criteria  OR risk > 0.65
    Soon      : ≥1 mild criterion  OR risk > 0.40
    Routine   : all criteria normal
    """
    if flags.urgent_criteria >= 3 or risk_score > 0.85:
        return 'Emergency (same-day referral)'
    if flags.urgent_criteria >= 1 or flags.alarm_criteria >= 2 or risk_score > 0.65:
        return 'Urgent (within 2 weeks)'
    if flags.positive_criteria >= 1 or risk_score > 0.40:
        return 'Soon (within 4–6 weeks)'
    return 'Routine (next scheduled visit)'


# ── RECOMMENDATION ENGINE ─────────────────────────────────────────────────────

def build_recommendations(
    flags: ClinicalFlags,
    risk_level: str,
    predicted_class: Optional[str] = None,
) -> List[str]:
    """
    Rule-based clinical action recommendations.
    Maps individual criterion grades → standard of care actions.
    """
    recs = []

    if risk_level == 'High' or flags.alarm_criteria >= 3:
        recs.append('Dermoscopic evaluation by a dermatologist is strongly recommended.')
        recs.append('Consider excisional biopsy for histopathological confirmation.')

    if flags.E_grade in ('Moderate', 'Severe'):
        recs.append('Document lesion timeline — photograph at 4-week intervals '
                    'to objectively track evolution.')

    if flags.D_grade in ('Moderate', 'Severe'):
        recs.append('Lesion diameter exceeds the 6mm clinical threshold — '
                    'measure precisely and record in patient notes.')

    if flags.C_grade in ('Moderate', 'Severe'):
        recs.append('Multi-colour pattern detected — exclude melanoma with full '
                    'body skin examination.')

    if flags.A_grade == 'Severe' and flags.B_grade == 'Severe':
        recs.append('Combined severe asymmetry and border irregularity is a '
                    'high-specificity melanoma indicator.')

    if predicted_class:
        cls = predicted_class.lower()
        if 'melanoma' in cls:
            recs.append('Classification suggests melanoma — immediate referral '
                        'and SLNB evaluation advised.')
        elif 'bcc' in cls or 'basal cell' in cls:
            recs.append('BCC suspected — surgical or Mohs excision consult recommended.')
        elif 'scc' in cls or 'squamous' in cls:
            recs.append('SCC suspected — full excision with clear margins advised.')

    if not recs:
        recs.append('Lesion appears low-risk. Routine skin check in 12 months.')

    return recs


# ── DIFFERENTIAL HINTS ────────────────────────────────────────────────────────

def build_differential_hints(flags: ClinicalFlags, A, B, C, D, E) -> List[str]:
    """
    Pattern-match ABCDE profile to likely differential diagnoses.
    These are hints for the VQA / RAG system — NOT definitive diagnoses.
    """
    hints = []

    # High A + high B + high C → melanoma profile
    if A > 0.6 and B > 0.5 and C > 0.5:
        hints.append('Profile consistent with melanoma (high ABС)')

    # Low A + low B + high E → evolving benign or early lesion
    if A < 0.4 and B < 0.4 and E > 0.5:
        hints.append('Symmetric but changing — consider seborrhoeic keratosis '
                     'or Spitz nevus')

    # High B + low A + low C → BCC profile
    if B > 0.6 and A < 0.4 and C < 0.35:
        hints.append('Border irregularity with relative symmetry — BCC possible')

    # High D alone → check acral lentiginous melanoma or large benign nevus
    if D > 0.7 and A < 0.4 and B < 0.4:
        hints.append('Large but symmetric lesion — congenital nevus vs. ALM')

    # All low → likely benign nevus or simple lesion
    if flags.positive_criteria == 0:
        hints.append('All criteria within normal range — benign profile')

    return hints


# ── MAIN INTERFACE ────────────────────────────────────────────────────────────

def interpret_abcde(
    result: ABCDEResult,
    predicted_class: Optional[str] = None,
) -> ClinicalReport:
    """
    Convert ABCDEResult → ClinicalReport.

    Parameters
    ----------
    result          : ABCDEResult from abcde_computation.compute_abcde()
    predicted_class : String from HC stage (e.g. 'Melanoma', 'BCC', 'Nevus')

    Returns
    -------
    ClinicalReport — fully structured for JSON serialisation and VQA input
    """
    A, B, C, D, E = result.A, result.B, result.C, result.D, result.E

    # Grade each criterion
    flags = ClinicalFlags(
        A_grade   = _grade(A, THRESHOLDS['A']),
        B_grade   = _grade(B, THRESHOLDS['B']),
        C_grade   = _grade(C, THRESHOLDS['C']),
        D_grade   = _grade(D, THRESHOLDS['D']),
        E_grade   = _grade(E, THRESHOLDS['E']),
        positive_criteria = sum(
            g != 'Normal'
            for g in [_grade(A, THRESHOLDS['A']), _grade(B, THRESHOLDS['B']),
                      _grade(C, THRESHOLDS['C']), _grade(D, THRESHOLDS['D']),
                      _grade(E, THRESHOLDS['E'])]
        ),
        alarm_criteria = sum(
            g in ('Moderate', 'Severe')
            for g in [_grade(A, THRESHOLDS['A']), _grade(B, THRESHOLDS['B']),
                      _grade(C, THRESHOLDS['C']), _grade(D, THRESHOLDS['D']),
                      _grade(E, THRESHOLDS['E'])]
        ),
        urgent_criteria = sum(
            g == 'Severe'
            for g in [_grade(A, THRESHOLDS['A']), _grade(B, THRESHOLDS['B']),
                      _grade(C, THRESHOLDS['C']), _grade(D, THRESHOLDS['D']),
                      _grade(E, THRESHOLDS['E'])]
        ),
    )

    urgency = classify_urgency(flags, result.risk_score)
    recommendations = build_recommendations(flags, result.risk_level, predicted_class)
    differentials    = build_differential_hints(flags, A, B, C, D, E)

    # ── Plain-English summary ──────────────────────────────────────────────
    positives = [c for c, g in [('Asymmetry', flags.A_grade), ('Border irregularity', flags.B_grade),
                                  ('Colour variation', flags.C_grade), ('Large diameter', flags.D_grade),
                                  ('Evolution', flags.E_grade)] if g != 'Normal']

    if not positives:
        summary = (f'The lesion shows no significant ABCDE flags. '
                   f'Overall risk score is {result.risk_score:.2f} ({result.risk_level}).')
    elif len(positives) <= 2:
        summary = (f'The lesion shows {" and ".join(positives).lower()} '
                   f'(risk score {result.risk_score:.2f}, {result.risk_level} risk). '
                   f'Clinical follow-up is advised.')
    else:
        summary = (f'Multiple ABCDE criteria flagged: {", ".join(positives).lower()}. '
                   f'Risk score is {result.risk_score:.2f} ({result.risk_level} risk). '
                   f'{urgency}.')

    return ClinicalReport(
        abcde_scores = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E},
        flags        = flags,
        risk_score   = result.risk_score,
        risk_level   = result.risk_level,
        urgency      = urgency,
        summary      = summary,
        recommendations    = recommendations,
        differential_hints = differentials,
        evidence           = result.evidence,
        raw_metrics        = result.raw_metrics,
    )


# ── FUSION FEATURE VECTOR ─────────────────────────────────────────────────────

def abcde_to_feature_vector(result: ABCDEResult) -> dict:
    """
    Converts ABCDE result into a flat feature dict that MARIA's
    fusion layer can consume as an additional modality.

    This is injected into the 'image_meta' or a dedicated 'abcde'
    modality tensor during IT Fusion.
    """
    report = interpret_abcde(result)
    flags  = report.flags

    grade_to_int = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}

    return {
        'abcde_A'          : result.A,
        'abcde_B'          : result.B,
        'abcde_C'          : result.C,
        'abcde_D'          : result.D,
        'abcde_E'          : result.E,
        'abcde_risk'       : result.risk_score,
        'abcde_A_grade'    : grade_to_int[flags.A_grade],
        'abcde_B_grade'    : grade_to_int[flags.B_grade],
        'abcde_C_grade'    : grade_to_int[flags.C_grade],
        'abcde_D_grade'    : grade_to_int[flags.D_grade],
        'abcde_E_grade'    : grade_to_int[flags.E_grade],
        'abcde_n_positive' : flags.positive_criteria,
        'abcde_n_alarm'    : flags.alarm_criteria,
    }


# ── DEMO ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Simulate a high-risk lesion result
    from abcde_computation import ABCDEResult

    mock_result = ABCDEResult(
        A=0.78, B=0.83, C=0.71, D=0.65, E=0.91,
        risk_score=0.87,
        risk_level='High',
        evidence=[
            'Asymmetric lesion (score 0.78)',
            'Irregular border (score 0.83) — compactness 2.8',
            'Multiple colours (score 0.71) — 4 clusters, ΔE 42.1',
            'Large diameter (score 0.65) — 7.8 mm',
            'Evolution reported (score 0.91) — growth, bleeding',
        ],
        raw_metrics={'A': {'h_iou': 0.51, 'v_iou': 0.43},
                     'B': {'compactness': 2.8, 'convexity_ratio': 0.72},
                     'C': {'active_color_clusters': 4, 'mean_delta_e': 42.1},
                     'D': {'max_diameter_mm': 7.8},
                     'E': {'matched_keywords': ['grew larger', 'bleeding'],
                           'raw_evolution_score': 0.72}}
    )

    report = interpret_abcde(mock_result, predicted_class='Melanoma')

    print('=== CLINICAL REPORT ===')
    print(f'Summary     : {report.summary}')
    print(f'Urgency     : {report.urgency}')
    print(f'A-Grade     : {report.flags.A_grade}')
    print(f'B-Grade     : {report.flags.B_grade}')
    print(f'C-Grade     : {report.flags.C_grade}')
    print(f'D-Grade     : {report.flags.D_grade}')
    print(f'E-Grade     : {report.flags.E_grade}')
    print(f'# Alarm     : {report.flags.alarm_criteria}')
    print('\nRecommendations:')
    for r in report.recommendations:
        print(f'  • {r}')
    print('\nDifferential hints:')
    for d in report.differential_hints:
        print(f'  • {d}')

    print('\nJSON output (for VQA/RAG):')
    print(report.to_json())

    print('\nFeature vector for MARIA:')
    print(abcde_to_feature_vector(mock_result))
