"""
json_builder.py
───────────────
Unified Clinical JSON builder for the IT Fusion stage.

Takes all upstream module outputs and assembles the final structured
clinical report JSON that is consumed by:
  • VQA module (for natural-language Q&A)
  • RAG retrieval pipeline
  • Frontend UI (for display)
  • Audit trail / EHR integration

All fields are consistently named and typed.
"""

import json
from typing import Optional, Dict, List, Any
from datetime import datetime


# ── SCHEMA VERSION ────────────────────────────────────────────────────────────
SCHEMA_VERSION = '1.3.0'


def build_json(
    # Core classification
    final_class: str,
    confidence: float,
    abcde: Dict[str, float],
    risk: float,
    explanation: List[str],

    # Extended fields (optional but recommended)
    abcde_grades: Optional[Dict[str, str]] = None,
    risk_level: str = 'Unknown',
    urgency: str = 'Unknown',

    # Classification details
    hc_class: Optional[str] = None,
    hc_confidence: Optional[float] = None,
    maria_class: Optional[str] = None,
    maria_confidence: Optional[float] = None,
    class_probs: Optional[Dict[str, float]] = None,

    # ABCDE details
    n_criteria_flagged: int = 0,
    flagged_criteria: Optional[List[str]] = None,
    alarm_criteria: Optional[List[str]] = None,

    # Treatment & drugs
    drug_recommendation: Optional[Dict] = None,
    drug_rule_applied: Optional[str] = None,

    # Clinical narrative
    clinical_summary: Optional[str] = None,
    differential_hints: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,

    # Patient metadata
    patient_age: Optional[float] = None,
    patient_gender: Optional[str] = None,
    patient_history: Optional[Dict[str, bool]] = None,

    # Dataset / source info
    image_path: Optional[str] = None,
    dataset_source: Optional[str] = None,
    modality: Optional[str] = None,

    # Raw metrics for provenance
    raw_metrics: Optional[Dict] = None,
    visual_hints: Optional[Dict] = None,
    confidence_note: Optional[str] = None,
    uncertainty_flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a fully structured Clinical JSON report.

    This is the canonical output format consumed by all downstream modules.

    Parameters
    ----------
    final_class      : Final predicted disease class (string)
    confidence       : Prediction confidence [0, 1]
    abcde            : Dict with A, B, C, D, E scores [0,1]
    risk             : Overall risk score [0,1]
    explanation      : List of evidence/finding strings
    ... (all other fields are optional enrichments)

    Returns
    -------
    dict — fully serialisable clinical JSON
    """

    # ── Timestamp ─────────────────────────────────────────────────────────────
    now = datetime.utcnow().isoformat() + 'Z'

    # ── Core classification block ──────────────────────────────────────────────
    classification_block = {
        'final_class'     : final_class,
        'confidence'      : round(float(confidence), 4),
    }
    if hc_class is not None:
        classification_block['hc_class']     = hc_class
        classification_block['hc_confidence'] = round(float(hc_confidence or 0), 4)
    if maria_class is not None:
        classification_block['maria_class']     = maria_class
        classification_block['maria_confidence'] = round(float(maria_confidence or 0), 4)
    if class_probs is not None:
        classification_block['class_probabilities'] = {
            k: round(float(v), 4) for k, v in class_probs.items()
        }

    # ── ABCDE block ───────────────────────────────────────────────────────────
    abcde_block = {
        'scores': {k: round(float(v), 4) for k, v in abcde.items()},
        'grades': abcde_grades or {},
        'n_criteria_flagged' : n_criteria_flagged,
        'flagged_criteria'   : flagged_criteria or [],
        'alarm_criteria'     : alarm_criteria or [],
    }

    # ── Risk block ────────────────────────────────────────────────────────────
    risk_block = {
        'score'          : round(float(risk), 4),
        'level'          : risk_level,
        'urgency'        : urgency,
    }

    # ── Patient block ─────────────────────────────────────────────────────────
    patient_block = {}
    if patient_age is not None and patient_age > 0:
        patient_block['age'] = float(patient_age)
    if patient_gender is not None:
        patient_block['gender'] = patient_gender
    if patient_history:
        patient_block['history'] = {
            k: bool(v) for k, v in patient_history.items()
        }

    # ── Source / provenance block ─────────────────────────────────────────────
    source_block = {}
    if image_path:
        source_block['image_path'] = image_path
    if dataset_source:
        source_block['dataset']  = dataset_source
    if modality:
        source_block['modality'] = modality

    # ── Treatment block ───────────────────────────────────────────────────────
    treatment_block = {}
    if drug_recommendation:
        treatment_block = {
            'surgical_referral'  : drug_recommendation.get('surgical_referral', False),
            'oncology_referral'  : drug_recommendation.get('oncology_referral', False),
            'options'            : drug_recommendation.get('options', []),
            'general_notes'      : drug_recommendation.get('general_notes', []),
            'disclaimer'         : drug_recommendation.get('disclaimer', _DISCLAIMER),
        }
        if drug_rule_applied:
            treatment_block['rule_applied'] = drug_rule_applied
    else:
        treatment_block['disclaimer'] = _DISCLAIMER

    # ── Assemble full report ──────────────────────────────────────────────────
    report = {
        '_schema_version'  : SCHEMA_VERSION,
        '_generated_at'    : now,

        'classification'   : classification_block,
        'ABCDE'            : abcde_block,
        'risk'             : risk_block,

        'clinical_summary' : clinical_summary or _auto_summary(
            final_class, risk_level, n_criteria_flagged, risk
        ),
        'evidence'         : explanation,
        'recommendations'  : recommendations or [],
        'differential_hints' : differential_hints or [],

        'treatment'        : treatment_block,

        'explainability'   : {
            'visual_hints'      : visual_hints or {},
            'confidence_note'   : confidence_note or '',
            'uncertainty_flags' : uncertainty_flags or [],
        },
    }

    if patient_block:
        report['patient'] = patient_block

    if source_block:
        report['source'] = source_block

    if raw_metrics:
        report['raw_metrics'] = raw_metrics

    return report


def build_json_from_modules(
    abcde_result,
    clinical_report,
    drug_recommendation,
    hc_class: str,
    hc_confidence: float,
    maria_class: Optional[str] = None,
    maria_confidence: Optional[float] = None,
    class_probs: Optional[Dict[str, float]] = None,
    explanation_report: Optional[Dict] = None,
    patient_age: Optional[float] = None,
    patient_gender: Optional[str] = None,
    patient_history: Optional[Dict] = None,
    image_path: Optional[str] = None,
    dataset_source: Optional[str] = None,
    modality: Optional[str] = None,
    uncertainty_flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience builder that takes dataclass outputs from the pipeline modules
    and assembles the unified Clinical JSON.

    Inputs follow the data classes from:
      abcde_result     ← ABCDEResult from abcde_computation.compute_abcde()
      clinical_report  ← ClinicalReport from abcde_inference.interpret_abcde()
      drug_recommendation ← DrugRecommendation from drug_rules.recommend_treatment()
      explanation_report  ← dict from explainability_generator.generate_explanation()
    """

    abcde_scores = {
        'A': abcde_result.A,
        'B': abcde_result.B,
        'C': abcde_result.C,
        'D': abcde_result.D,
        'E': abcde_result.E,
    }

    abcde_grades = {
        'A': clinical_report.flags.A_grade,
        'B': clinical_report.flags.B_grade,
        'C': clinical_report.flags.C_grade,
        'D': clinical_report.flags.D_grade,
        'E': clinical_report.flags.E_grade,
    }

    flagged = [c for c, g in abcde_grades.items() if g != 'Normal']
    alarm   = [c for c, g in abcde_grades.items() if g in ('Moderate', 'Severe')]

    drug_dict = None
    if drug_recommendation:
        from dataclasses import asdict
        drug_dict = {
            'surgical_referral': drug_recommendation.surgical_referral,
            'oncology_referral': drug_recommendation.oncology_referral,
            'general_notes'    : drug_recommendation.general_notes,
            'disclaimer'       : drug_recommendation.disclaimer,
            'options'          : [
                {
                    'name'      : o.name,
                    'route'     : o.route,
                    'line'      : o.line,
                    'indication': o.indication,
                    'reference' : o.reference,
                    'contraindications': o.contraindications,
                }
                for o in drug_recommendation.options
            ],
        }

    expl = explanation_report or {}

    return build_json(
        final_class         = maria_class or hc_class,
        confidence          = maria_confidence or hc_confidence,
        abcde               = abcde_scores,
        risk                = abcde_result.risk_score,
        explanation         = abcde_result.evidence,

        abcde_grades        = abcde_grades,
        risk_level          = abcde_result.risk_level,
        urgency             = clinical_report.urgency,

        hc_class            = hc_class,
        hc_confidence       = hc_confidence,
        maria_class         = maria_class,
        maria_confidence    = maria_confidence,
        class_probs         = class_probs,

        n_criteria_flagged  = clinical_report.flags.positive_criteria,
        flagged_criteria    = flagged,
        alarm_criteria      = alarm,

        drug_recommendation = drug_dict,

        clinical_summary    = expl.get('narrative') or clinical_report.summary,
        differential_hints  = clinical_report.differential_hints,
        recommendations     = clinical_report.recommendations,

        patient_age         = patient_age,
        patient_gender      = patient_gender,
        patient_history     = patient_history,

        image_path          = image_path,
        dataset_source      = dataset_source,
        modality            = modality,

        raw_metrics         = abcde_result.raw_metrics,
        visual_hints        = expl.get('visual_hints'),
        confidence_note     = expl.get('confidence_note'),
        uncertainty_flags   = uncertainty_flags or [],
    )


# ── JSON UTILITIES ────────────────────────────────────────────────────────────

def to_json_string(report: Dict, indent: int = 2) -> str:
    """Serialise clinical report to JSON string."""
    return json.dumps(report, indent=indent, default=str)


def save_json(report: Dict, path: str, indent: int = 2) -> None:
    """Save clinical report to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=indent, default=str)
    print(f'[json_builder] Saved report to {path}')


def load_json(path: str) -> Dict:
    """Load a previously saved clinical report."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── PRIVATE HELPERS ───────────────────────────────────────────────────────────

_DISCLAIMER = (
    'This report is generated by an AI-assisted clinical decision support system. '
    'It is intended for informational purposes only and must NOT replace the '
    'professional judgement of a qualified dermatologist or physician. '
    'All treatment decisions must be made following full clinical assessment.'
)


def _auto_summary(final_class: str, risk_level: str,
                  n_flagged: int, risk_score: float) -> str:
    if n_flagged == 0:
        return (
            f'Automated analysis predicts {final_class} with no significant ABCDE flags '
            f'(risk score {risk_score:.2f}, {risk_level} risk). '
            f'Routine monitoring recommended.'
        )
    elif n_flagged <= 2:
        return (
            f'{n_flagged} ABCDE {"criterion" if n_flagged == 1 else "criteria"} flagged. '
            f'Predicted class: {final_class}. Risk score: {risk_score:.2f} ({risk_level}). '
            f'Clinical follow-up advised.'
        )
    else:
        return (
            f'Multiple ABCDE criteria flagged ({n_flagged}/5). '
            f'Predicted class: {final_class}. Risk score: {risk_score:.2f} ({risk_level}). '
            f'Urgent dermatologist review recommended.'
        )


# ── DEMO ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    report = build_json(
        final_class  = 'Melanoma',
        confidence   = 0.87,
        abcde        = {'A': 0.72, 'B': 0.65, 'C': 0.58, 'D': 0.44, 'E': 0.91},
        risk         = 0.74,
        explanation  = [
            'Asymmetric lesion (score 0.72)',
            'Irregular border (score 0.65)',
            'Multiple colours (score 0.58)',
            'Evolution reported (score 0.91) — grew, bleeding',
        ],
        abcde_grades    = {'A': 'Severe', 'B': 'Moderate', 'C': 'Moderate', 'D': 'Mild', 'E': 'Severe'},
        risk_level      = 'High',
        urgency         = 'Urgent (within 2 weeks)',
        hc_class        = 'Melanoma',
        hc_confidence   = 0.83,
        maria_class     = 'MEL',
        maria_confidence= 0.87,
        n_criteria_flagged = 4,
        flagged_criteria   = ['A', 'B', 'C', 'E'],
        alarm_criteria     = ['A', 'B', 'C', 'E'],
        recommendations = [
            'Dermoscopic evaluation by a dermatologist is strongly recommended.',
            'Consider excisional biopsy for histopathological confirmation.',
        ],
        differential_hints = [
            'Profile consistent with melanoma (high ABC)',
        ],
        drug_recommendation = {
            'surgical_referral': True,
            'oncology_referral': True,
            'general_notes': ['Wide local excision is the primary treatment.'],
            'disclaimer': _DISCLAIMER,
            'options': [
                {
                    'name': 'Wide Local Excision',
                    'route': 'surgical',
                    'line': 'first',
                    'indication': 'Primary treatment',
                    'reference': 'NCCN Melanoma 2024',
                    'contraindications': [],
                }
            ],
        },
        patient_age   = 52.0,
        patient_gender= 'Female',
        dataset_source= 'PAD-UFES-20',
        image_path    = 'PAT_1516_1765_530.png',
    )

    print(to_json_string(report))
