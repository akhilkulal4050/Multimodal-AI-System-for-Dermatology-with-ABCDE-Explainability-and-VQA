"""
explainability_generator.py
───────────────────────────
Generates structured, human-readable clinical explanations from
ABCDE scores + MARIA classification output.

Inputs:
  • abcde          : dict with keys A, B, C, D, E (floats 0–1)
  • abcde_grades   : dict with grades 'Normal'|'Mild'|'Moderate'|'Severe'
  • risk_score     : float [0,1]
  • risk_level     : str 'Low'|'Moderate'|'High'
  • predicted_class: str from HC/MARIA
  • confidence     : float [0,1]
  • raw_metrics    : dict of intermediate computation values

Outputs:
  • Plain-English narrative explanation (for VQA / RAG / UI)
  • Structured evidence list (for JSON clinical report)
  • Visual heat map keywords (for highlighting in image UI)
  • Confidence-calibrated disclaimer
"""

from typing import Optional, Dict, List, Any


# ── THRESHOLD DEFINITIONS ─────────────────────────────────────────────────────

_GRADE_ORDER = ['Normal', 'Mild', 'Moderate', 'Severe']

_CRITERION_DESCRIPTIONS = {
    'A': {
        'full':  'Asymmetry',
        'Normal':   'The lesion is largely symmetric in both axes.',
        'Mild':     'Slight asymmetry detected — one axis shows minor imbalance.',
        'Moderate': 'Moderate asymmetry: the lesion differs noticeably between halves.',
        'Severe':   'Significant asymmetry in both axes — a key melanoma indicator.',
    },
    'B': {
        'full':  'Border',
        'Normal':   'The lesion border is smooth and well-defined.',
        'Mild':     'Slightly irregular border with minor notching or fading.',
        'Moderate': 'Irregular, poorly defined border — consider dermoscopic assessment.',
        'Severe':   'Highly irregular, broken, or satellite border pattern detected.',
    },
    'C': {
        'full':  'Colour Variation',
        'Normal':   'Uniform colour distribution within the lesion.',
        'Mild':     'Minor colour variation — 1–2 tones present.',
        'Moderate': 'Multiple colour zones detected (brown, black, grey, pink).',
        'Severe':   'Complex multi-colour pattern (≥4 tones) — high risk indicator.',
    },
    'D': {
        'full':  'Diameter',
        'Normal':   'Lesion diameter is below the 6 mm clinical threshold.',
        'Mild':     'Lesion approaching 6 mm — monitor closely.',
        'Moderate': 'Lesion diameter exceeds 6 mm — document and measure precisely.',
        'Severe':   'Large lesion (>10 mm) — clinical measurement and imaging advised.',
    },
    'E': {
        'full':  'Evolution',
        'Normal':   'No significant recent change reported in the lesion.',
        'Mild':     'Minor change noted — recommend 4–6 week photographic follow-up.',
        'Moderate': 'Notable evolution: size, colour, or shape change reported.',
        'Severe':   'Rapid or significant change — bleeding, ulceration, or fast growth.',
    },
}


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def _grade_above(grade: str, threshold: str) -> bool:
    """Return True if grade severity >= threshold severity."""
    return _GRADE_ORDER.index(grade) >= _GRADE_ORDER.index(threshold)


def _format_score(value: float) -> str:
    return f'{value:.2f}'


# ── MAIN EXPLANATION GENERATOR ────────────────────────────────────────────────

def generate_explanation(
    abcde: Dict[str, float],
    abcde_grades: Optional[Dict[str, str]] = None,
    risk_score: float = 0.0,
    risk_level: str = 'Low',
    predicted_class: Optional[str] = None,
    confidence: float = 0.0,
    raw_metrics: Optional[Dict[str, Any]] = None,
    patient_age: Optional[float] = None,
    patient_gender: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full explainability report generator.

    Parameters
    ----------
    abcde          : {'A': 0.72, 'B': 0.61, 'C': 0.54, 'D': 0.40, 'E': 0.88}
    abcde_grades   : {'A': 'Severe', 'B': 'Moderate', ...}  (auto-computed if None)
    risk_score     : overall weighted risk [0,1]
    risk_level     : 'Low' | 'Moderate' | 'High'
    predicted_class: HC/MARIA classification string
    confidence     : model confidence [0,1]
    raw_metrics    : intermediate computation metrics from abcde_computation
    patient_age    : patient age in years (optional)
    patient_gender : 'Male' | 'Female' | None

    Returns
    -------
    dict with:
      - 'narrative'      : plain-English paragraph
      - 'evidence_list'  : list of individual criterion findings
      - 'flagged_criteria': list of criterion names that are >= Mild
      - 'visual_hints'   : UI-level hints for overlay display
      - 'confidence_note': calibrated reliability statement
      - 'structured'     : machine-readable per-criterion breakdown
    """

    A = abcde.get('A', 0.0)
    B = abcde.get('B', 0.0)
    C = abcde.get('C', 0.0)
    D = abcde.get('D', 0.0)
    E = abcde.get('E', 0.0)

    # Auto-compute grades if not provided
    if abcde_grades is None:
        abcde_grades = _auto_grade(A, B, C, D, E)

    raw = raw_metrics or {}

    # ── 1. Evidence per criterion ─────────────────────────────────────────────
    evidence_list = []

    for crit, val in [('A', A), ('B', B), ('C', C), ('D', D), ('E', E)]:
        grade = abcde_grades.get(crit, 'Normal')
        desc = _CRITERION_DESCRIPTIONS[crit]
        finding = _build_criterion_finding(crit, val, grade, desc, raw.get(crit, {}))
        evidence_list.append(finding)

    # ── 2. Flagged criteria ───────────────────────────────────────────────────
    flagged = [c for c in 'ABCDE'
               if _grade_above(abcde_grades.get(c, 'Normal'), 'Mild')]
    alarm   = [c for c in 'ABCDE'
               if _grade_above(abcde_grades.get(c, 'Normal'), 'Moderate')]

    # ── 3. Narrative paragraph ────────────────────────────────────────────────
    narrative = _build_narrative(
        A, B, C, D, E,
        abcde_grades, flagged, alarm,
        risk_score, risk_level,
        predicted_class, confidence,
        patient_age, patient_gender,
        raw,
    )

    # ── 4. Visual hints for UI overlay ───────────────────────────────────────
    visual_hints = _build_visual_hints(A, B, C, D, E, abcde_grades, raw)

    # ── 5. Confidence note ────────────────────────────────────────────────────
    confidence_note = _build_confidence_note(confidence, risk_level, len(flagged))

    # ── 6. Structured machine-readable output ─────────────────────────────────
    structured = {
        c: {
            'score'       : round(abcde.get(c, 0.0), 4),
            'grade'       : abcde_grades.get(c, 'Normal'),
            'full_name'   : _CRITERION_DESCRIPTIONS[c]['full'],
            'description' : _CRITERION_DESCRIPTIONS[c].get(abcde_grades.get(c, 'Normal'), ''),
            'flagged'     : c in flagged,
        }
        for c in 'ABCDE'
    }

    return {
        'narrative'        : narrative,
        'evidence_list'    : evidence_list,
        'flagged_criteria' : flagged,
        'alarm_criteria'   : alarm,
        'visual_hints'     : visual_hints,
        'confidence_note'  : confidence_note,
        'structured'       : structured,
        'risk_score'       : round(risk_score, 4),
        'risk_level'       : risk_level,
        'predicted_class'  : predicted_class,
    }


# ── PER-CRITERION FINDING BUILDER ─────────────────────────────────────────────

def _build_criterion_finding(
    crit: str, val: float, grade: str, desc: dict, raw: dict
) -> str:
    base = f'{desc["full"]} (score {_format_score(val)}, {grade}): {desc.get(grade, "")}'

    # Append raw metric details where available
    if crit == 'A' and raw:
        h_iou = raw.get('h_iou', None)
        v_iou = raw.get('v_iou', None)
        if h_iou is not None:
            base += f' [H-IoU={h_iou:.2f}, V-IoU={v_iou:.2f}]'

    elif crit == 'B' and raw:
        compact = raw.get('compactness', None)
        convex  = raw.get('convexity_ratio', None)
        if compact is not None:
            base += f' [compactness={compact:.2f}, convexity={convex:.2f}]'

    elif crit == 'C' and raw:
        n_clust = raw.get('active_color_clusters', None)
        delta_e = raw.get('mean_delta_e', None)
        if n_clust is not None:
            base += f' [{n_clust} dominant colour clusters, ΔE={delta_e:.1f}]'

    elif crit == 'D' and raw:
        max_d = raw.get('max_diameter_mm', None)
        if max_d is not None:
            base += f' [measured {max_d:.1f} mm]'

    elif crit == 'E' and raw:
        kws = raw.get('matched_keywords', [])
        boost = raw.get('keyword_boost', 0.0)
        if kws:
            base += f' [keywords: {", ".join(kws[:3])}; boost +{boost:.2f}]'

    return base


# ── NARRATIVE BUILDER ─────────────────────────────────────────────────────────

def _build_narrative(
    A, B, C, D, E,
    grades, flagged, alarm,
    risk_score, risk_level,
    predicted_class, confidence,
    patient_age, patient_gender,
    raw,
) -> str:
    parts = []

    # Patient context
    ctx_parts = []
    if patient_age and patient_age > 0:
        ctx_parts.append(f'{int(patient_age)}-year-old')
    if patient_gender:
        ctx_parts.append(patient_gender.lower())
    ctx = ' '.join(ctx_parts) if ctx_parts else 'the'
    patient_str = f'the {ctx} patient' if ctx_parts else 'the patient'

    # Opening
    if not flagged:
        parts.append(
            f'Analysis of the lesion belonging to {patient_str} reveals no significant '
            f'ABCDE criteria above the normal threshold. '
            f'The overall risk score is {risk_score:.2f}, classified as {risk_level} risk.'
        )
    else:
        n_flagged = len(flagged)
        criterion_str = ', '.join(
            [_CRITERION_DESCRIPTIONS[c]['full'] for c in flagged]
        )
        parts.append(
            f'Lesion analysis for {patient_str} flagged {n_flagged} ABCDE '
            f'{"criterion" if n_flagged == 1 else "criteria"}: {criterion_str}. '
            f'Overall risk score is {risk_score:.2f} ({risk_level} risk).'
        )

    # Per-criterion narrative contributions (only flagged ones)
    for crit in flagged:
        val = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}[crit]
        grade = grades.get(crit, 'Normal')
        desc = _CRITERION_DESCRIPTIONS[crit].get(grade, '')
        parts.append(f'{_CRITERION_DESCRIPTIONS[crit]["full"]}: {desc}')

    # Classification
    if predicted_class:
        conf_str = f'{confidence:.1%}' if confidence else ''
        conf_note = f' (confidence: {conf_str})' if conf_str else ''
        parts.append(
            f'The classification model predicts {predicted_class}{conf_note}.'
        )

    # High-risk combined pattern
    if 'A' in alarm and 'B' in alarm:
        parts.append(
            'Combined asymmetry and border irregularity at alarm level is a '
            'high-specificity pattern for melanoma — specialist referral is strongly advised.'
        )

    if C > 0.65:
        n_clust = raw.get('C', {}).get('active_color_clusters', '?')
        parts.append(
            f'The colour complexity ({n_clust} dominant clusters) further supports '
            'the need for dermoscopic evaluation.'
        )

    if E > 0.6:
        kws = raw.get('E', {}).get('matched_keywords', [])
        if kws:
            kw_str = ', '.join(kws[:3])
            parts.append(
                f'The lesion shows documented evolution ({kw_str}), '
                'which warrants urgent clinical attention.'
            )

    return ' '.join(parts)


# ── VISUAL HINTS ──────────────────────────────────────────────────────────────

def _build_visual_hints(A, B, C, D, E, grades, raw) -> Dict[str, Any]:
    """
    Returns hints for the front-end to overlay visual indicators on the image.
    Keys map to colour/intensity guidance for the UI layer.
    """
    hints = {}

    grade_to_color = {
        'Normal':   '#4ade80',   # green
        'Mild':     '#facc15',   # yellow
        'Moderate': '#fb923c',   # orange
        'Severe':   '#ef4444',   # red
    }

    for crit, val in [('A', A), ('B', B), ('C', C), ('D', D), ('E', E)]:
        grade = grades.get(crit, 'Normal')
        hints[crit] = {
            'color'      : grade_to_color[grade],
            'intensity'  : round(val, 3),
            'label'      : _CRITERION_DESCRIPTIONS[crit]['full'],
            'grade'      : grade,
            'show_overlay': _grade_above(grade, 'Mild'),
        }

    # Bounding box suggestion based on D score
    max_d = raw.get('D', {}).get('max_diameter_mm', None)
    if max_d is not None:
        hints['diameter_mm'] = round(max_d, 2)
        hints['show_size_ring'] = max_d > 6.0

    # Colour cluster overlay suggestion
    n_clust = raw.get('C', {}).get('active_color_clusters', 0)
    hints['color_clusters'] = n_clust
    hints['show_color_overlay'] = n_clust > 2

    return hints


# ── CONFIDENCE NOTE ───────────────────────────────────────────────────────────

def _build_confidence_note(confidence: float, risk_level: str, n_flagged: int) -> str:
    if confidence >= 0.85:
        reliability = 'high-confidence'
    elif confidence >= 0.65:
        reliability = 'moderate-confidence'
    else:
        reliability = 'low-confidence'

    note = (
        f'This is a {reliability} ({confidence:.1%}) automated analysis. '
    )

    if risk_level == 'High' or n_flagged >= 3:
        note += (
            'Given the elevated risk profile, results should be reviewed by a '
            'qualified dermatologist before any clinical decision.'
        )
    elif risk_level == 'Moderate':
        note += (
            'Dermatologist review is recommended, especially for the flagged criteria.'
        )
    else:
        note += (
            'Routine clinical follow-up is advised as per standard skin monitoring guidelines.'
        )

    return note


# ── AUTO-GRADER ───────────────────────────────────────────────────────────────

def _auto_grade(A, B, C, D, E) -> Dict[str, str]:
    """Grade each criterion using default ABCDE thresholds."""
    THRESHOLDS = {
        'A': (0.30, 0.55, 0.75),
        'B': (0.30, 0.55, 0.75),
        'C': (0.25, 0.50, 0.70),
        'D': (0.35, 0.55, 0.80),
        'E': (0.20, 0.45, 0.70),
    }
    def _grade(v, t):
        mild, mod, sev = t
        if v >= sev:   return 'Severe'
        if v >= mod:   return 'Moderate'
        if v >= mild:  return 'Mild'
        return 'Normal'

    return {
        'A': _grade(A, THRESHOLDS['A']),
        'B': _grade(B, THRESHOLDS['B']),
        'C': _grade(C, THRESHOLDS['C']),
        'D': _grade(D, THRESHOLDS['D']),
        'E': _grade(E, THRESHOLDS['E']),
    }


# ── DEMO ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json

    result = generate_explanation(
        abcde={'A': 0.72, 'B': 0.65, 'C': 0.58, 'D': 0.44, 'E': 0.91},
        risk_score=0.74,
        risk_level='High',
        predicted_class='Melanoma',
        confidence=0.87,
        raw_metrics={
            'A': {'h_iou': 0.48, 'v_iou': 0.41},
            'B': {'compactness': 2.9, 'convexity_ratio': 0.68},
            'C': {'active_color_clusters': 4, 'mean_delta_e': 38.5},
            'D': {'max_diameter_mm': 7.3},
            'E': {'matched_keywords': ['grew larger', 'bleeding'],
                  'keyword_boost': 0.10},
        },
        patient_age=52.0,
        patient_gender='Female',
    )

    print('=== EXPLAINABILITY REPORT ===\n')
    print('Narrative:')
    print(result['narrative'])
    print('\nEvidence List:')
    for e in result['evidence_list']:
        print(f'  • {e}')
    print('\nFlagged:', result['flagged_criteria'])
    print('Alarm  :', result['alarm_criteria'])
    print('\nVisual Hints (A):', result['visual_hints']['A'])
    print('\nConfidence Note:')
    print(result['confidence_note'])
    print('\nFull Structured JSON:')
    print(json.dumps(result['structured'], indent=2))
