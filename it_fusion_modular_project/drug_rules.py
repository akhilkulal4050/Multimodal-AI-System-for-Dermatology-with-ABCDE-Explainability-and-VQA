"""
drug_rules.py
─────────────
Rule-based drug / treatment recommendation engine for the IT Fusion stage.

This is intentionally NOT a neural model — drug recommendations in a
clinical support system should be:
  (a) Auditable — every recommendation traces to a named rule
  (b) Conservative — the system suggests, a clinician decides
  (c) Safe — contraindications are always checked first

Approach: Condition → Treatment Line lookup table with:
  • First-line / second-line options
  • Contraindication flags based on patient history
  • Topical vs systemic tier
  • Reference (guideline source)

IMPORTANT: This module produces *suggestions for clinical discussion only*.
It must NOT be used as a standalone prescribing tool.

Conditions covered (matching HC + IT Fusion classification outputs):
  Oncology:
    Melanoma, BCC, SCC, AKIEC (Actinic Keratosis), DF, VASC
  Benign / common:
    Nevus, BKL / Seborrhoeic Keratosis
  Inflammatory:
    Psoriasis, Eczema / Atopic Dermatitis, Lichen Planus, Drug Eruption,
    Urticaria, Bullous Pemphigoid, Pityriasis Rosea
  Infectious:
    Tinea, Herpes Zoster, Herpes Simplex, Impetigo, Cellulitis,
    Scabies, Molluscum Contagiosum, Folliculitis
  Hair & Pigment:
    Alopecia Areata, Vitiligo, Rosacea, Acne Vulgaris,
    Allergic Contact Dermatitis, Keloid
  Genetic:
    Epidermolysis Bullosa
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import json


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────

@dataclass
class TreatmentOption:
    name: str
    route: str             # 'topical' | 'systemic' | 'surgical' | 'procedural'
    line: str              # 'first' | 'second' | 'adjuvant' | 'palliative'
    indication: str
    contraindications: List[str] = field(default_factory=list)
    reference: str = ''


@dataclass
class DrugRecommendation:
    condition: str
    risk_level: str
    options: List[TreatmentOption]
    surgical_referral: bool
    oncology_referral: bool
    general_notes: List[str]
    disclaimer: str = (
        'These are clinical decision-support suggestions only. '
        'All treatment decisions must be made by a qualified dermatologist '
        'or physician following full clinical assessment.'
    )

    def to_dict(self):
        return {
            'condition': self.condition,
            'risk_level': self.risk_level,
            'surgical_referral': self.surgical_referral,
            'oncology_referral': self.oncology_referral,
            'options': [asdict(o) for o in self.options],
            'general_notes': self.general_notes,
            'disclaimer': self.disclaimer,
        }

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)


# ── TREATMENT KNOWLEDGE BASE ──────────────────────────────────────────────────
# Sources: AAD, EADO, NCCN, BAD, EDF, WHO guidelines (generic drug names only)

TREATMENT_KB: Dict[str, dict] = {

    # ── ONCOLOGY ──────────────────────────────────────────────────────────────

    'melanoma': {
        'surgical_referral': True,
        'oncology_referral': True,
        'notes': [
            'Wide local excision is the primary treatment (margins per Breslow thickness).',
            'Sentinel lymph node biopsy recommended for T1b+.',
            'Adjuvant immunotherapy (anti-PD-1) for Stage III/IV.',
            'BRAF testing mandatory before targeted therapy.',
        ],
        'options': [
            TreatmentOption('Wide Local Excision', 'surgical', 'first',
                            'Primary treatment; margins 0.5–2 cm per Breslow',
                            reference='NCCN Melanoma 2024'),
            TreatmentOption('Pembrolizumab', 'systemic', 'adjuvant',
                            'Anti-PD-1 immunotherapy for Stage III/IV resected',
                            contraindications=['active autoimmune disease',
                                              'immunosuppressive therapy',
                                              'organ transplant'],
                            reference='KEYNOTE-054'),
            TreatmentOption('Nivolumab', 'systemic', 'adjuvant',
                            'Anti-PD-1; alternative to pembrolizumab',
                            contraindications=['active autoimmune disease',
                                              'organ transplant'],
                            reference='CheckMate 238'),
            TreatmentOption('Dabrafenib + Trametinib', 'systemic', 'first',
                            'BRAF/MEK inhibitors for BRAF V600E/K mutant melanoma only',
                            contraindications=['braf_wildtype',
                                              'hepatic_impairment'],
                            reference='COMBI-v/COMBI-d'),
        ],
    },

    'bcc': {
        'surgical_referral': True,
        'oncology_referral': False,
        'notes': [
            'Mohs micrographic surgery is gold standard for high-risk BCC.',
            'Topical imiquimod or 5-FU for superficial low-risk BCC.',
            'Hedgehog pathway inhibitors for unresectable/metastatic BCC.',
        ],
        'options': [
            TreatmentOption('Mohs Micrographic Surgery', 'surgical', 'first',
                            'High-risk / facial BCC; highest cure rate',
                            reference='AAD BCC Guideline 2023'),
            TreatmentOption('Standard Excision', 'surgical', 'first',
                            'Non-facial, low-risk BCC; 4mm margins',
                            reference='AAD BCC Guideline 2023'),
            TreatmentOption('Imiquimod 5% cream', 'topical', 'first',
                            'Superficial BCC only; 5×/week for 6 weeks',
                            contraindications=['nodular BCC', 'immunosuppression'],
                            reference='AAD 2023'),
            TreatmentOption('Fluorouracil 5% cream', 'topical', 'second',
                            'Superficial BCC; alternative to imiquimod',
                            contraindications=['nodular or infiltrative BCC',
                                              'pregnancy', 'renal_impairment'],
                            reference='AAD 2023'),
            TreatmentOption('Vismodegib', 'systemic', 'second',
                            'Hedgehog inhibitor for locally advanced/metastatic BCC',
                            contraindications=['pregnancy (teratogenic)',
                                              'hepatic_impairment'],
                            reference='ERIVANCE trial'),
        ],
    },

    'scc': {
        'surgical_referral': True,
        'oncology_referral': False,
        'notes': [
            'Excision with 4–6mm margins for low-risk SCC.',
            'Mohs surgery for high-risk locations (head, neck, genitals).',
            'Cemiplimab for metastatic/locally advanced SCC.',
        ],
        'options': [
            TreatmentOption('Standard Excision', 'surgical', 'first',
                            'Low-risk cSCC; 4–6mm margins',
                            reference='AAD cSCC Guideline 2023'),
            TreatmentOption('Mohs Micrographic Surgery', 'surgical', 'first',
                            'High-risk cSCC; facial/genital locations',
                            reference='AAD 2023'),
            TreatmentOption('Cemiplimab', 'systemic', 'first',
                            'Anti-PD-1 for metastatic/locally advanced cSCC',
                            contraindications=['active autoimmune disease',
                                              'organ transplant'],
                            reference='EMPOWER-CSCC-1'),
            TreatmentOption('Radiotherapy', 'procedural', 'adjuvant',
                            'Post-op adjuvant or inoperable SCC',
                            reference='NCCN 2024'),
        ],
    },

    'akiec': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Actinic keratosis (AK) is a pre-malignant lesion — treat to prevent SCC.',
            'Single lesions → cryotherapy; field disease → topical therapy.',
            'Emphasise sun protection (SPF 50+, protective clothing).',
        ],
        'options': [
            TreatmentOption('Cryotherapy (liquid nitrogen)', 'procedural', 'first',
                            'Single or few lesions; 1–2 freeze-thaw cycles',
                            reference='EDF AK Guideline 2022'),
            TreatmentOption('Fluorouracil 5% cream', 'topical', 'first',
                            'Field therapy; twice daily for 2–4 weeks',
                            contraindications=['pregnancy', 'renal_impairment'],
                            reference='EDF 2022'),
            TreatmentOption('Imiquimod 3.75% or 5% cream', 'topical', 'first',
                            'Field therapy; 2–3×/week for 12–16 weeks',
                            contraindications=['immunosuppression'],
                            reference='EDF 2022'),
            TreatmentOption('Diclofenac 3% gel', 'topical', 'second',
                            'Twice daily for 60–90 days; mild efficacy, well tolerated',
                            reference='EDF 2022'),
            TreatmentOption('Photodynamic Therapy (PDT)', 'procedural', 'first',
                            'Aminolevulinic acid + red light; effective for field AK',
                            reference='EDF 2022'),
            TreatmentOption('Tirbanibulin 1% ointment', 'topical', 'second',
                            'Once daily for 5 days; face/scalp AK',
                            reference='PIVOTAL trial 2020'),
        ],
    },

    'df': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Dermatofibroma is benign; treatment is cosmetic unless symptomatic.',
            'Reassurance is the primary intervention for asymptomatic lesions.',
        ],
        'options': [
            TreatmentOption('Observation / Reassurance', 'procedural', 'first',
                            'Asymptomatic lesions — no treatment required',
                            reference='Clinical practice'),
            TreatmentOption('Shave excision or cryotherapy', 'procedural', 'second',
                            'Cosmetic removal; high recurrence with shave',
                            reference='Clinical practice'),
            TreatmentOption('Excisional biopsy', 'surgical', 'second',
                            'Atypical or enlarging DF to exclude DFSP',
                            reference='Clinical practice'),
        ],
    },

    'vasc': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Vascular lesions include haemangioma, pyogenic granuloma, angiokeratoma.',
            'Treatment depends on subtype, size, and symptoms.',
        ],
        'options': [
            TreatmentOption('Pulsed Dye Laser (PDL)', 'procedural', 'first',
                            'Port wine stains, haemangiomas, spider naevi',
                            contraindications=['pacemaker (caution)'],
                            reference='AAD Vascular Lesions 2023'),
            TreatmentOption('Timolol 0.5% gel', 'topical', 'first',
                            'Superficial infantile haemangioma',
                            contraindications=['asthma', 'heart block',
                                              'bradycardia'],
                            reference='Pediatric Dermatology guidelines'),
            TreatmentOption('Propranolol (oral)', 'systemic', 'first',
                            'Infantile haemangioma requiring systemic therapy',
                            contraindications=['asthma', 'hypoglycaemia',
                                              'bradycardia', 'heart_failure'],
                            reference='HEMANGEOL label / EDF 2023'),
            TreatmentOption('Shave / Curette', 'surgical', 'first',
                            'Pyogenic granuloma — curettage with cautery',
                            reference='Clinical practice'),
        ],
    },

    # ── BENIGN / KERATOSES ────────────────────────────────────────────────────

    'nevus': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Benign melanocytic nevus — no treatment necessary unless atypia.',
            'Regular self-examination and annual dermatologist review advised.',
        ],
        'options': [
            TreatmentOption('Observation / Monitoring', 'procedural', 'first',
                            'Annual review; photograph for baseline',
                            reference='AAD Nevus Guidelines'),
            TreatmentOption('Excision biopsy', 'surgical', 'second',
                            'Indicated if dysplasia suspected or rapid change',
                            reference='AAD 2023'),
        ],
    },

    'bkl': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Seborrhoeic keratosis / benign keratosis is entirely benign.',
            'Treatment is cosmetic only — no malignant potential.',
            'Rapid appearance of multiple lesions (Leser-Trélat sign) warrants '
            'investigation for internal malignancy.',
        ],
        'options': [
            TreatmentOption('Observation / Reassurance', 'procedural', 'first',
                            'Asymptomatic lesions — no treatment required',
                            reference='AAD 2023'),
            TreatmentOption('Cryotherapy (liquid nitrogen)', 'procedural', 'second',
                            'Cosmetic removal; 1–2 freeze-thaw cycles',
                            reference='Clinical practice'),
            TreatmentOption('Shave excision / curettage', 'surgical', 'second',
                            'Cosmetic removal; low recurrence',
                            reference='Clinical practice'),
            TreatmentOption('Hydrogen peroxide 40% solution', 'topical', 'second',
                            'FDA-approved (Eskata) for raised SK; 2 applications',
                            reference='FINAL-1/FINAL-2 trials 2019'),
        ],
    },

    # ── INFLAMMATORY ──────────────────────────────────────────────────────────

    'psoriasis': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Chronic inflammatory disease — treatment goal is remission, not cure.',
            'Mild (BSA <3%) → topical; moderate-severe → systemic or biologic.',
            'Screen for psoriatic arthritis — rheumatology co-management if present.',
            'Metabolic syndrome monitoring recommended for systemic therapy patients.',
        ],
        'options': [
            TreatmentOption('Topical corticosteroids (moderate–potent)', 'topical', 'first',
                            'Mild-moderate plaque psoriasis; twice daily',
                            contraindications=['rosacea', 'periorbital skin',
                                              'prolonged use on face/flexures'],
                            reference='BAD Psoriasis Guideline 2023'),
            TreatmentOption('Calcipotriol (Vitamin D analogue)', 'topical', 'first',
                            'Combination with steroid (calcipotriol/betamethasone) preferred',
                            contraindications=['hypercalcaemia', 'renal_impairment'],
                            reference='BAD 2023'),
            TreatmentOption('Methotrexate', 'systemic', 'first',
                            'Moderate-severe psoriasis; weekly dosing + folic acid',
                            contraindications=['pregnancy', 'hepatic_impairment',
                                              'renal_impairment', 'immunosuppression'],
                            reference='BAD 2023 / NICE TA'),
            TreatmentOption('Ciclosporin', 'systemic', 'first',
                            'Short-term rapid control; max 2 years continuous use',
                            contraindications=['renal_impairment', 'hypertension',
                                              'immunosuppression', 'malignancy_history'],
                            reference='BAD 2023'),
            TreatmentOption('Adalimumab', 'systemic', 'second',
                            'Anti-TNF biologic for moderate-severe psoriasis',
                            contraindications=['active TB', 'active infection',
                                              'heart_failure', 'demyelinating disease'],
                            reference='NICE TA498 2018'),
            TreatmentOption('Secukinumab', 'systemic', 'second',
                            'Anti-IL-17A; rapid onset, high efficacy',
                            contraindications=['active Crohn\'s disease',
                                              'active infection'],
                            reference='NICE TA350 / ERASURE trial'),
            TreatmentOption('Ixekizumab', 'systemic', 'second',
                            'Anti-IL-17A; alternative to secukinumab',
                            contraindications=['active infection',
                                              'inflammatory bowel disease'],
                            reference='UNCOVER trials'),
            TreatmentOption('Narrowband UVB Phototherapy', 'procedural', 'first',
                            'Moderate psoriasis; 3×/week; safe in pregnancy',
                            reference='BAD 2023'),
        ],
    },

    'eczema': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Atopic dermatitis — identify and avoid triggers.',
            'Emollients are the cornerstone of all severity levels.',
            'Wet wrapping for acute severe flares in children.',
            'Avoid contact allergens — patch testing if contact dermatitis suspected.',
        ],
        'options': [
            TreatmentOption('Emollients (liberal, frequent use)', 'topical', 'first',
                            'All severity levels; apply 2–3× daily minimum',
                            reference='NICE CG57 / BAD AD Guideline 2023'),
            TreatmentOption('Topical corticosteroids (mild–moderate)', 'topical', 'first',
                            'Flare control; use lowest effective potency',
                            contraindications=['prolonged face use',
                                              'skin infections (bacterial/fungal)'],
                            reference='BAD 2023'),
            TreatmentOption('Topical calcineurin inhibitors (tacrolimus/pimecrolimus)',
                            'topical', 'first',
                            'Steroid-sparing; face and flexures; maintenance',
                            contraindications=['active skin infection',
                                              'immunosuppression',
                                              'malignancy_history'],
                            reference='BAD 2023'),
            TreatmentOption('Dupilumab', 'systemic', 'second',
                            'Anti-IL-4Rα biologic for moderate-severe AD ≥6 years',
                            contraindications=['active helminth infection'],
                            reference='LIBERTY AD SOLO / NICE TA534 2018'),
            TreatmentOption('Tralokinumab', 'systemic', 'second',
                            'Anti-IL-13 biologic; ≥18 years moderate-severe AD',
                            contraindications=['active infection'],
                            reference='ECZTRA trials / NICE TA 2022'),
            TreatmentOption('Ciclosporin', 'systemic', 'second',
                            'Short-term systemic for severe refractory AD',
                            contraindications=['renal_impairment', 'hypertension',
                                              'immunosuppression'],
                            reference='BAD 2023'),
            TreatmentOption('Narrowband UVB Phototherapy', 'procedural', 'second',
                            'Moderate-severe AD unresponsive to topicals',
                            reference='BAD 2023'),
        ],
    },

    'allergic_contact_dermatitis': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Identify and eliminate the causative allergen — patch testing essential.',
            'Avoidance of trigger is the definitive treatment.',
            'Occupational exposure: involve occupational health.',
        ],
        'options': [
            TreatmentOption('Allergen avoidance', 'procedural', 'first',
                            'Definitive treatment — identify trigger via patch testing',
                            reference='ESCD Contact Dermatitis Guideline 2023'),
            TreatmentOption('Topical corticosteroids (moderate–potent)', 'topical', 'first',
                            'Acute and chronic phase; taper over 2–3 weeks',
                            contraindications=['skin infection'],
                            reference='ESCD 2023'),
            TreatmentOption('Topical calcineurin inhibitors', 'topical', 'second',
                            'Steroid-sparing; face and flexures',
                            contraindications=['active skin infection'],
                            reference='ESCD 2023'),
            TreatmentOption('Oral prednisolone (short course)', 'systemic', 'second',
                            'Severe widespread contact dermatitis; 7–14 days tapering',
                            contraindications=['diabetes', 'osteoporosis',
                                              'active peptic ulcer',
                                              'immunosuppression'],
                            reference='Clinical practice'),
        ],
    },

    'urticaria': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Acute urticaria (<6 weeks) — identify and avoid trigger if possible.',
            'Chronic spontaneous urticaria (>6 weeks) — stepwise approach.',
            'Angioedema with anaphylaxis risk: ensure patient has adrenaline auto-injector.',
        ],
        'options': [
            TreatmentOption('Non-sedating antihistamine (cetirizine / loratadine / fexofenadine)',
                            'systemic', 'first',
                            'Daily dosing; up-dose to 4× standard if needed',
                            contraindications=['hepatic_impairment (fexofenadine)'],
                            reference='EAACI/GA2LEN Urticaria Guidelines 2022'),
            TreatmentOption('Omalizumab (anti-IgE)', 'systemic', 'second',
                            'Chronic spontaneous urticaria refractory to antihistamines',
                            contraindications=['active parasitic infection'],
                            reference='ASTERIA I/II / NICE TA339 2015'),
            TreatmentOption('Short-course oral corticosteroid', 'systemic', 'adjuvant',
                            'Acute severe urticaria; max 10 days',
                            contraindications=['diabetes', 'osteoporosis',
                                              'active peptic ulcer'],
                            reference='EAACI 2022'),
        ],
    },

    'lichen_planus': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Lichen planus is self-limiting in most cases (skin LP resolves in 1–2 years).',
            'Oral LP has malignant potential — 6-monthly review of oral lesions.',
            'Hepatitis C association — screen if risk factors present.',
        ],
        'options': [
            TreatmentOption('Potent topical corticosteroid (clobetasol)', 'topical', 'first',
                            'Skin and oral LP; twice daily for 4–8 weeks',
                            contraindications=['skin infection'],
                            reference='BAD LP Guideline 2023'),
            TreatmentOption('Topical calcineurin inhibitors (tacrolimus)', 'topical', 'first',
                            'Oral and genital LP; steroid-sparing',
                            contraindications=['active infection',
                                              'immunosuppression'],
                            reference='BAD 2023'),
            TreatmentOption('Oral prednisolone', 'systemic', 'second',
                            'Widespread or erosive LP; short tapering course',
                            contraindications=['diabetes', 'osteoporosis',
                                              'immunosuppression'],
                            reference='BAD 2023'),
            TreatmentOption('Hydroxychloroquine', 'systemic', 'second',
                            'Chronic LP — monitor for retinopathy',
                            contraindications=['retinal disease', 'G6PD deficiency'],
                            reference='BAD 2023'),
        ],
    },

    'drug_eruption': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Identify and stop the causative drug immediately.',
            'Severe reactions (SJS, TEN, DRESS) require hospital admission.',
            'Allergy documentation in patient records is essential.',
        ],
        'options': [
            TreatmentOption('Causative drug withdrawal', 'procedural', 'first',
                            'Mandatory — identify offending drug and stop',
                            reference='Clinical practice'),
            TreatmentOption('Topical corticosteroid + emollient', 'topical', 'first',
                            'Mild maculopapular eruption for symptomatic relief',
                            reference='Clinical practice'),
            TreatmentOption('Oral antihistamine (cetirizine)', 'systemic', 'first',
                            'Pruritus management in mild-moderate eruption',
                            reference='Clinical practice'),
            TreatmentOption('Systemic corticosteroid', 'systemic', 'second',
                            'Severe hypersensitivity / DRESS — prednisolone 0.5–1 mg/kg',
                            contraindications=['active infection', 'diabetes',
                                              'immunosuppression'],
                            reference='Clinical practice / EDF SJS guidelines'),
        ],
    },

    'bullous_pemphigoid': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Autoimmune blistering disease — most common in elderly.',
            'Confirm diagnosis with perilesional biopsy + DIF and serology (anti-BP180/BP230).',
            'Tetracycline + nicotinamide combination for mild disease is well tolerated.',
        ],
        'options': [
            TreatmentOption('Ultrapotent topical corticosteroid (clobetasol propionate)',
                            'topical', 'first',
                            'First-line for mild-moderate BP — apply to active lesions',
                            reference='BAD BP Guideline 2023'),
            TreatmentOption('Doxycycline + nicotinamide', 'systemic', 'first',
                            'Mild-moderate BP; better safety than systemic steroids in elderly',
                            contraindications=['pregnancy', 'hepatic_impairment'],
                            reference='BLISTER trial 2017'),
            TreatmentOption('Oral prednisolone', 'systemic', 'second',
                            'Moderate-severe BP; 0.5 mg/kg/day with taper',
                            contraindications=['diabetes', 'osteoporosis',
                                              'active infection',
                                              'immunosuppression'],
                            reference='BAD 2023'),
            TreatmentOption('Rituximab', 'systemic', 'second',
                            'Refractory BP — anti-CD20 therapy',
                            contraindications=['active infection',
                                              'hepatitis B (reactivation risk)',
                                              'immunosuppression'],
                            reference='BAD 2023'),
        ],
    },

    'pityriasis_rosea': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Self-limiting viral exanthem — typically resolves in 6–12 weeks.',
            'Reassurance is the primary intervention.',
            'Aciclovir may shorten duration if started early.',
        ],
        'options': [
            TreatmentOption('Reassurance and observation', 'procedural', 'first',
                            'Self-limiting; resolves spontaneously in 6–12 weeks',
                            reference='BAD Pityriasis Rosea Guideline'),
            TreatmentOption('Emollient + mild topical corticosteroid', 'topical', 'first',
                            'Symptomatic itch relief',
                            reference='Clinical practice'),
            TreatmentOption('Oral antihistamine (cetirizine / loratadine)', 'systemic', 'first',
                            'Pruritus management',
                            reference='Clinical practice'),
            TreatmentOption('Aciclovir (oral)', 'systemic', 'second',
                            'Early severe disease; 400mg 5×/day for 7 days',
                            contraindications=['renal_impairment (dose adjustment)'],
                            reference='BAD 2023'),
        ],
    },

    # ── INFECTIOUS ────────────────────────────────────────────────────────────

    'tinea': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Confirm diagnosis with skin scraping + microscopy / culture before systemic therapy.',
            'Treat contacts and footwear (tinea pedis) to prevent reinfection.',
            'Tinea capitis (scalp) requires systemic antifungal — topical alone insufficient.',
        ],
        'options': [
            TreatmentOption('Terbinafine 1% cream', 'topical', 'first',
                            'Tinea corporis/cruris/pedis; once daily 1–2 weeks',
                            reference='BAD Dermatophyte Guideline 2014'),
            TreatmentOption('Clotrimazole 1% cream', 'topical', 'first',
                            'Alternative topical; twice daily for 4 weeks',
                            reference='BAD 2014'),
            TreatmentOption('Oral terbinafine', 'systemic', 'first',
                            'Tinea capitis, onychomycosis, or extensive disease; '
                            '250mg daily 6 weeks (fingernails) / 12 weeks (toenails)',
                            contraindications=['hepatic_impairment',
                                              'autoimmune hepatitis history'],
                            reference='BAD 2014'),
            TreatmentOption('Oral itraconazole', 'systemic', 'second',
                            'Alternative systemic; useful for tinea versicolor',
                            contraindications=['hepatic_impairment',
                                              'heart_failure', 'warfarin/coumadin'],
                            reference='BAD 2014'),
            TreatmentOption('Oral griseofulvin', 'systemic', 'first',
                            'Tinea capitis in children (preferred); 10–20 mg/kg/day',
                            contraindications=['pregnancy', 'hepatic_impairment',
                                              'porphyria'],
                            reference='BAD 2014'),
        ],
    },

    'herpes_zoster': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Start antiviral within 72h of rash onset for maximum benefit.',
            'Ophthalmic zoster (V1) — urgent ophthalmology referral.',
            'Post-herpetic neuralgia (PHN) prevention — key benefit of early treatment.',
            'Consider shingles vaccine (Shingrix) for prevention in ≥50 years.',
        ],
        'options': [
            TreatmentOption('Valaciclovir', 'systemic', 'first',
                            '1000mg TDS for 7 days; better bioavailability than aciclovir',
                            contraindications=['renal_impairment (dose reduction)'],
                            reference='BAD Herpes Zoster Guideline 2023'),
            TreatmentOption('Aciclovir (oral)', 'systemic', 'first',
                            '800mg 5×/day for 7 days; if valaciclovir unavailable',
                            contraindications=['renal_impairment (dose reduction)'],
                            reference='BAD 2023'),
            TreatmentOption('IV Aciclovir', 'systemic', 'first',
                            'Immunocompromised patients; 10mg/kg TDS for 7–10 days',
                            contraindications=['renal_impairment (dose adjustment)'],
                            reference='BAD 2023'),
            TreatmentOption('Amitriptyline / gabapentin / pregabalin', 'systemic', 'adjuvant',
                            'Post-herpetic neuralgia pain management',
                            contraindications=['heart block', 'urinary retention'],
                            reference='NICE PHN guidelines'),
        ],
    },

    'herpes_simplex': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Recurrent orolabial HSV: episodic or suppressive therapy based on frequency.',
            'Genital HSV: refer to GUM/sexual health clinic for contact tracing.',
            'Eczema herpeticum: urgent IV aciclovir and dermatology/infectious disease review.',
        ],
        'options': [
            TreatmentOption('Topical aciclovir 5% cream', 'topical', 'first',
                            'Orolabial HSV — apply at prodrome; reduces duration modestly',
                            reference='BAD HSV Guideline'),
            TreatmentOption('Oral aciclovir', 'systemic', 'first',
                            'Episodic: 200–400mg 5×/day for 5 days; start at prodrome',
                            contraindications=['renal_impairment (dose reduction)'],
                            reference='BAD HSV Guideline'),
            TreatmentOption('Valaciclovir (suppressive)', 'systemic', 'first',
                            '500mg daily for suppression if ≥6 recurrences/year',
                            contraindications=['renal_impairment (dose reduction)'],
                            reference='BAD HSV / IHMF guidelines'),
        ],
    },

    'impetigo': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Highly contagious — school exclusion until lesions healed or 48h after antibiotic.',
            'Swab for culture and sensitivities if treatment failure or recurrence.',
            'MRSA-positive: use topical mupirocin or systemic agent per sensitivities.',
        ],
        'options': [
            TreatmentOption('Hydrogen peroxide 1% cream (Crystacide)', 'topical', 'first',
                            'Non-antibiotic first-line; apply 2–3× daily for 7–10 days',
                            reference='NICE NG153 2020 / BAD Impetigo Guideline'),
            TreatmentOption('Topical mupirocin 2% ointment', 'topical', 'first',
                            'Limited localised disease; TDS for 5 days',
                            contraindications=['mupirocin-resistant MRSA'],
                            reference='NICE NG153 2020'),
            TreatmentOption('Topical fusidic acid 2% cream', 'topical', 'first',
                            'Localised non-bullous impetigo; TDS for 5 days',
                            contraindications=['fusidic acid resistance'],
                            reference='NICE NG153 2020'),
            TreatmentOption('Oral flucloxacillin', 'systemic', 'first',
                            'Extensive disease or failed topical; 500mg QDS for 7 days',
                            contraindications=['penicillin allergy'],
                            reference='NICE NG153 2020'),
            TreatmentOption('Oral cefalexin', 'systemic', 'second',
                            'Penicillin-allergic patients (non-anaphylaxis); 500mg BD-TDS',
                            contraindications=['cephalosporin allergy'],
                            reference='NICE NG153 2020'),
        ],
    },

    'cellulitis': {
        'surgical_referral': True,
        'oncology_referral': False,
        'notes': [
            'Mark erythema border with pen to monitor progression.',
            'Hospital admission if systemic features, rapid spread, or immunocompromised.',
            'Treat predisposing factors: tinea pedis, leg oedema, skin breaks.',
            'Recurrent cellulitis: consider prophylactic low-dose penicillin.',
        ],
        'options': [
            TreatmentOption('Oral flucloxacillin', 'systemic', 'first',
                            'Non-facial, non-severe cellulitis; 500mg QDS for 5–7 days',
                            contraindications=['penicillin allergy',
                                              'hepatic_impairment'],
                            reference='NICE NG141 2019 / Eron classification'),
            TreatmentOption('Oral cefalexin', 'systemic', 'first',
                            'Penicillin allergy (non-anaphylactic); 500mg QDS',
                            contraindications=['cephalosporin allergy'],
                            reference='NICE NG141 2019'),
            TreatmentOption('Oral clarithromycin', 'systemic', 'second',
                            'Penicillin anaphylaxis history; 500mg BD for 5–7 days',
                            contraindications=['hepatic_impairment',
                                              'QT prolongation', 'warfarin/coumadin'],
                            reference='NICE NG141 2019'),
            TreatmentOption('IV benzylpenicillin / flucloxacillin', 'systemic', 'first',
                            'Severe / hospitalised cellulitis — IV therapy',
                            contraindications=['penicillin allergy'],
                            reference='NICE NG141 2019'),
        ],
    },

    'scabies': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Treat all close contacts simultaneously — single treatment failure common.',
            'Norwegian (crusted) scabies: multiple treatments + isolation precautions.',
            'Decontaminate clothing, bedding (60°C wash or sealed bag 72h).',
        ],
        'options': [
            TreatmentOption('Permethrin 5% cream', 'topical', 'first',
                            'Apply from neck down, leave 8–12h, wash off; repeat at 1 week',
                            contraindications=['allergy to chrysanthemums (pyrethroids)'],
                            reference='WHO / BAD Scabies Guideline 2023'),
            TreatmentOption('Malathion 0.5% aqueous liquid', 'topical', 'second',
                            'Alternative to permethrin; apply as above',
                            reference='BAD 2023'),
            TreatmentOption('Oral ivermectin', 'systemic', 'second',
                            '200 mcg/kg single dose; repeat at 2 weeks; '
                            'crusted scabies requires multiple doses',
                            contraindications=['pregnancy', 'weight <15 kg',
                                              'hepatic_impairment'],
                            reference='WHO / BAD 2023'),
        ],
    },

    'molluscum': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Molluscum contagiosum is self-limiting — typically resolves in 6–18 months.',
            'Treatment is usually cosmetic or to prevent autoinoculation.',
            'Immunocompromised patients may require more aggressive treatment.',
        ],
        'options': [
            TreatmentOption('Observation / Watchful waiting', 'procedural', 'first',
                            'Self-limiting; no treatment required for healthy children',
                            reference='BAD Molluscum Guideline'),
            TreatmentOption('Cantharidin 0.7% solution (blistering agent)', 'topical', 'first',
                            'Apply briefly in clinic; effective but causes blister',
                            contraindications=['immunosuppression (caution)'],
                            reference='BAD 2023'),
            TreatmentOption('Cryotherapy (liquid nitrogen)', 'procedural', 'second',
                            'Office-based; effective but painful in children',
                            reference='BAD 2023'),
            TreatmentOption('Potassium hydroxide 5–10% solution', 'topical', 'second',
                            'Self-applied twice daily until lesions resolve',
                            reference='Clinical practice'),
        ],
    },

    'folliculitis': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Identify type: bacterial (most common), fungal (Malassezia), hot tub (Pseudomonas).',
            'Shaving technique modification and skin hygiene advice.',
            'Recurrent folliculitis: nasal MRSA carriage swab — consider decolonisation.',
        ],
        'options': [
            TreatmentOption('Topical mupirocin 2% ointment', 'topical', 'first',
                            'Localised bacterial folliculitis; TDS for 5 days',
                            reference='Clinical practice'),
            TreatmentOption('Topical clindamycin 1% gel', 'topical', 'first',
                            'Moderate folliculitis; BD for 4–6 weeks',
                            reference='Clinical practice'),
            TreatmentOption('Oral flucloxacillin', 'systemic', 'second',
                            'Extensive or recurrent bacterial folliculitis; '
                            '500mg QDS for 5–7 days',
                            contraindications=['penicillin allergy',
                                              'hepatic_impairment'],
                            reference='Clinical practice'),
            TreatmentOption('Topical ketoconazole 2% shampoo / cream', 'topical', 'first',
                            'Malassezia (fungal) folliculitis; leave on 5 min, wash off',
                            reference='Clinical practice'),
        ],
    },

    # ── HAIR & PIGMENT ────────────────────────────────────────────────────────

    'acne': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Stepwise approach based on severity (mild/moderate/severe).',
            'Isotretinoin: mandatory pregnancy prevention programme (iPLEDGE/PREG).',
            'Avoid prolonged topical antibiotic monotherapy — antibiotic resistance risk.',
            'Results take 6–12 weeks — manage patient expectations.',
        ],
        'options': [
            TreatmentOption('Topical adapalene + benzoyl peroxide (Epiduo)', 'topical', 'first',
                            'Mild-moderate acne; once daily at night',
                            contraindications=['pregnancy (adapalene)'],
                            reference='BAD Acne Guideline 2021 / NICE NG198'),
            TreatmentOption('Topical benzoyl peroxide 2.5–5%', 'topical', 'first',
                            'Mild acne; anti-comedonal + antibacterial',
                            reference='BAD 2021'),
            TreatmentOption('Topical clindamycin + benzoyl peroxide', 'topical', 'first',
                            'Inflammatory mild-moderate acne; avoid monotherapy',
                            reference='BAD 2021'),
            TreatmentOption('Oral doxycycline', 'systemic', 'first',
                            'Moderate-severe acne; 100mg daily for 3–6 months',
                            contraindications=['pregnancy', 'age <12 years'],
                            reference='BAD 2021 / NICE NG198'),
            TreatmentOption('Oral lymecycline', 'systemic', 'first',
                            'Alternative tetracycline; 408mg daily for 3–6 months',
                            contraindications=['pregnancy', 'age <12 years'],
                            reference='BAD 2021'),
            TreatmentOption('Combined oral contraceptive (co-cyprindiol)', 'systemic', 'second',
                            'Female patients with hormonal acne; after 3rd-line failure',
                            contraindications=['VTE history', 'smoker >35 years',
                                              'migraine with aura'],
                            reference='BAD 2021'),
            TreatmentOption('Oral isotretinoin', 'systemic', 'second',
                            'Severe/scarring acne; 0.5–1 mg/kg/day; '
                            'mandatory pregnancy prevention programme',
                            contraindications=['pregnancy', 'hepatic_impairment',
                                              'hyperlipidaemia', 'depression history'],
                            reference='BAD 2021 / EMA SmPC'),
        ],
    },

    'rosacea': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Identify and advise avoidance of triggers (UV, heat, alcohol, spicy food).',
            'Subtype-specific treatment: papulopustular vs erythematotelangiectatic vs phymatous.',
            'Sun protection is essential (SPF 30+ daily).',
            'Ocular rosacea: ophthalmology co-management.',
        ],
        'options': [
            TreatmentOption('Topical metronidazole 0.75–1% gel/cream', 'topical', 'first',
                            'Papulopustular rosacea; once or twice daily',
                            reference='BAD Rosacea Guideline 2021'),
            TreatmentOption('Topical azelaic acid 15% gel', 'topical', 'first',
                            'Papulopustular rosacea; twice daily',
                            reference='BAD 2021'),
            TreatmentOption('Topical ivermectin 1% cream', 'topical', 'first',
                            'Superior efficacy for inflammatory rosacea; once daily',
                            reference='ATTRACT/EMERGE trials / BAD 2021'),
            TreatmentOption('Topical brimonidine 0.33% gel', 'topical', 'first',
                            'Persistent facial erythema; apply once daily (not papules)',
                            contraindications=['cardiovascular disease (bradycardia risk)'],
                            reference='BAD 2021'),
            TreatmentOption('Oral doxycycline 40mg (subantimicrobial)', 'systemic', 'first',
                            'Anti-inflammatory dose; 40mg modified-release daily',
                            contraindications=['pregnancy', 'age <12'],
                            reference='BAD 2021 / NICE NG201'),
            TreatmentOption('Pulsed dye laser / IPL', 'procedural', 'second',
                            'Erythematotelangiectatic rosacea; telangiectasia',
                            contraindications=['pacemaker (caution)'],
                            reference='BAD 2021'),
        ],
    },

    'vitiligo': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Psychological impact is significant — offer counselling support.',
            'Sun protection essential for depigmented areas (no melanin protection).',
            'Active disease vs stable disease affects treatment choice.',
            'Rule out associated autoimmune conditions (thyroid, diabetes, pernicious anaemia).',
        ],
        'options': [
            TreatmentOption('Topical corticosteroid (potent)', 'topical', 'first',
                            'Active/limited vitiligo; once daily 3–4 months then review',
                            contraindications=['face (use calcineurin inhibitors instead)',
                                              'prolonged use on flexures'],
                            reference='BAD Vitiligo Guideline 2023'),
            TreatmentOption('Topical tacrolimus 0.1% ointment', 'topical', 'first',
                            'Face and flexures; steroid-sparing; twice daily',
                            contraindications=['active infection',
                                              'immunosuppression'],
                            reference='BAD 2023'),
            TreatmentOption('Narrowband UVB phototherapy (NB-UVB)', 'procedural', 'first',
                            'Widespread active vitiligo; 2–3×/week for 12–24 months',
                            reference='BAD 2023'),
            TreatmentOption('Ruxolitinib 1.5% cream (JAK1/2 inhibitor)', 'topical', 'second',
                            'Non-segmental facial vitiligo ≥12 years; '
                            'first topical approved specifically for vitiligo (FDA 2022)',
                            contraindications=['active serious infection'],
                            reference='TRuE-V trials / FDA approval 2022'),
        ],
    },

    'alopecia_areata': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Alopecia totalis/universalis: lower response rate to treatment.',
            'Spontaneous remission occurs in ~50% of limited patch AA within 1 year.',
            'Nail changes and ophiasis pattern = poorer prognosis.',
            'Screen for associated autoimmune disease (thyroid, coeliac).',
        ],
        'options': [
            TreatmentOption('Intralesional corticosteroid (triamcinolone acetonide)',
                            'procedural', 'first',
                            'Patchy AA; injected into scalp; every 4–6 weeks',
                            contraindications=['active scalp infection'],
                            reference='BAD Alopecia Areata Guideline 2023'),
            TreatmentOption('Potent topical corticosteroid', 'topical', 'first',
                            'Patchy AA; once or twice daily for 3 months',
                            reference='BAD 2023'),
            TreatmentOption('Baricitinib (JAK1/2 inhibitor)', 'systemic', 'second',
                            'Severe AA (≥50% scalp hair loss); 4mg daily',
                            contraindications=['active infection', 'immunosuppression',
                                              'renal_impairment', 'VTE history',
                                              'malignancy_history'],
                            reference='BRAVE-AA1/AA2 / FDA approval 2022'),
            TreatmentOption('Ritlecitinib (JAK3/TEC inhibitor)', 'systemic', 'second',
                            'Severe AA ≥12 years; 50mg daily',
                            contraindications=['active infection', 'immunosuppression',
                                              'hepatic_impairment'],
                            reference='ALLEGRO trials / FDA approval 2023'),
            TreatmentOption('Topical minoxidil 5%', 'topical', 'adjuvant',
                            'Adjunct to stimulate regrowth; once or twice daily',
                            contraindications=['scalp dermatitis', 'pregnancy'],
                            reference='Clinical practice'),
        ],
    },

    'keloid': {
        'surgical_referral': True,
        'oncology_referral': False,
        'notes': [
            'Keloids recur after surgery alone — always combine with adjuvant therapy.',
            'Prevention: minimise tension on wounds; use silicone gel/sheets post-injury.',
            'High-risk sites: sternum, shoulders, earlobes, deltoid.',
        ],
        'options': [
            TreatmentOption('Intralesional triamcinolone acetonide', 'procedural', 'first',
                            '10–40 mg/mL; every 4–6 weeks; flattens lesion',
                            reference='BAD Keloid Guideline 2023'),
            TreatmentOption('Silicone gel sheeting', 'topical', 'first',
                            'Daily application ≥12h/day for 3–6 months; prevention',
                            reference='BAD 2023'),
            TreatmentOption('Cryotherapy (intralesional / contact)', 'procedural', 'second',
                            'Combined with steroid injection; multiple sessions',
                            reference='BAD 2023'),
            TreatmentOption('Surgical excision + intralesional steroid', 'surgical', 'second',
                            'Refractory lesions — always combine to reduce recurrence',
                            contraindications=['immunosuppression',
                                              'active wound infection'],
                            reference='BAD 2023'),
            TreatmentOption('Post-op radiotherapy', 'procedural', 'adjuvant',
                            'High-risk/large keloids; adjuvant to surgery',
                            reference='BAD 2023'),
        ],
    },

    # ── GENETIC ───────────────────────────────────────────────────────────────

    'epidermolysis_bullosa': {
        'surgical_referral': True,
        'oncology_referral': False,
        'notes': [
            'Epidermolysis Bullosa (EB) is a rare genetic blistering disorder.',
            'Treatment is multidisciplinary — wound care, pain management, nutrition.',
            'Dystrophic EB: monitor for SCC transformation (annual biopsy of chronic wounds).',
        ],
        'options': [
            TreatmentOption('Wound dressings (non-adherent)', 'topical', 'first',
                            'Mepitel, Mepilex, Polymem — minimise trauma',
                            reference='DEBRA EB Care Guidelines 2023'),
            TreatmentOption('Topical antiseptics (chlorhexidine)', 'topical', 'first',
                            'Infected wounds; avoid iodine-containing products',
                            reference='DEBRA 2023'),
            TreatmentOption('Birch bark extract (Filsuvez gel)', 'topical', 'first',
                            'First approved topical for EB wound healing (EU/US 2023)',
                            reference='EASE trial 2023 / EMA approval'),
            TreatmentOption('Systemic corticosteroids', 'systemic', 'second',
                            'JEB with significant inflammation — short course only',
                            contraindications=['DEB (limited evidence)', 'diabetes',
                                              'osteoporosis', 'active infection'],
                            reference='DEBRA 2023'),
        ],
    },

    # ── FALLBACK ──────────────────────────────────────────────────────────────

    'default': {
        'surgical_referral': False,
        'oncology_referral': False,
        'notes': [
            'Condition-specific treatment data not found in knowledge base.',
            'Please consult a dermatologist for evaluation.',
        ],
        'options': [
            TreatmentOption('Dermatologist Referral', 'procedural', 'first',
                            'Unclassified condition — specialist assessment required',
                            reference='Standard of care'),
        ],
    },
}


# ── CONDITION NAME NORMALISER ─────────────────────────────────────────────────

_ALIAS_MAP = {
    # Oncology
    'mel': 'melanoma', 'melanoma': 'melanoma',
    'bcc': 'bcc', 'basal cell carcinoma': 'bcc', 'basal cell': 'bcc',
    'scc': 'scc', 'squamous cell carcinoma': 'scc', 'squamous cell': 'scc',
    'akiec': 'akiec', 'actinic keratosis': 'akiec', 'ack': 'akiec',
    'intraepithelial carcinoma': 'akiec',
    'df': 'df', 'dermatofibroma': 'df',
    'vasc': 'vasc', 'vascular': 'vasc', 'haemangioma': 'vasc',
    'hemangioma': 'vasc',
    # Benign keratoses
    'nv': 'nevus', 'nev': 'nevus', 'nevus': 'nevus', 'mole': 'nevus',
    'naevus': 'nevus', 'melanocytic nevus': 'nevus',
    'bkl': 'bkl', 'seborrhoeic keratosis': 'bkl', 'seborrheic keratosis': 'bkl',
    'sek': 'bkl', 'benign keratosis': 'bkl', 'solar lentigo': 'bkl',
    'lichen planus-like keratosis': 'bkl',
    # Inflammatory
    'psoriasis': 'psoriasis',
    'eczema': 'eczema', 'atopic dermatitis': 'eczema', 'ad': 'eczema',
    'allergic contact dermatitis': 'allergic_contact_dermatitis',
    'contact dermatitis': 'allergic_contact_dermatitis',
    'urticaria': 'urticaria', 'hives': 'urticaria',
    'lichen planus': 'lichen_planus', 'lp': 'lichen_planus',
    'drug eruption': 'drug_eruption', 'drug reaction': 'drug_eruption',
    'bullous pemphigoid': 'bullous_pemphigoid', 'bp': 'bullous_pemphigoid',
    'pityriasis rosea': 'pityriasis_rosea',
    # Infectious
    'tinea': 'tinea', 'ringworm': 'tinea', 'dermatophytosis': 'tinea',
    'tinea corporis': 'tinea', 'tinea pedis': 'tinea',
    'tinea capitis': 'tinea', 'onychomycosis': 'tinea',
    'herpes zoster': 'herpes_zoster', 'shingles': 'herpes_zoster',
    'varicella zoster': 'herpes_zoster',
    'herpes simplex': 'herpes_simplex', 'cold sore': 'herpes_simplex',
    'hsv': 'herpes_simplex',
    'impetigo': 'impetigo',
    'cellulitis': 'cellulitis', 'erysipelas': 'cellulitis',
    'scabies': 'scabies',
    'molluscum': 'molluscum', 'molluscum contagiosum': 'molluscum',
    'folliculitis': 'folliculitis',
    # Hair & Pigment
    'acne': 'acne', 'acne vulgaris': 'acne',
    'rosacea': 'rosacea',
    'vitiligo': 'vitiligo',
    'alopecia areata': 'alopecia_areata', 'alopecia': 'alopecia_areata',
    'keloid': 'keloid', 'hypertrophic scar': 'keloid',
    # Genetic
    'eb': 'epidermolysis_bullosa',
    'epidermolysis bullosa': 'epidermolysis_bullosa',
    'dystrophic epidermolysis bullosa': 'epidermolysis_bullosa',
}


def _normalise_condition(raw: str) -> str:
    key = raw.strip().lower()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    # Partial match fallback
    for alias, target in _ALIAS_MAP.items():
        if alias in key:
            return target
    return 'default'


# ── CONTRAINDICATION CHECKER ──────────────────────────────────────────────────

def _check_contraindications(
    option: TreatmentOption,
    patient_history: Optional[dict] = None,
) -> List[str]:
    """
    Returns a list of triggered contraindication warnings.

    patient_history keys (all boolean unless noted):
      asthma, autoimmune, immunosuppressed, pregnant,
      transplant, heart_failure, renal_impairment, hepatic_impairment,
      braf_wildtype, g6pd_deficiency, warfarin, pacemaker,
      diabetes, osteoporosis, vte_history, malignancy_history,
      active_infection, active_tb, inflammatory_bowel_disease
    """
    if not patient_history or not option.contraindications:
        return []

    warnings = []
    ph = {k.lower(): v for k, v in patient_history.items()}

    CI_CHECKS = [
        ('asthma',                    'asthma',                'contraindicated in asthma'),
        ('autoimmune',                'autoimmune',            'use caution in autoimmune disease'),
        ('immunosuppres',             'immunosuppressed',      'avoid in immunosuppressed patients'),
        ('pregnan',                   'pregnant',              'CONTRAINDICATED in pregnancy'),
        ('transplant',                'transplant',            'PD-1 inhibitors may cause graft rejection'),
        ('heart',                     'heart_failure',         'caution in heart failure / bradycardia'),
        ('renal_impairment',          'renal_impairment',      'dose adjustment required in renal impairment'),
        ('hepatic_impairment',        'hepatic_impairment',    'avoid or reduce dose in hepatic impairment'),
        ('braf_wildtype',             'braf_wildtype',         'BRAF/MEK inhibitors require BRAF V600E/K mutation'),
        ('g6pd',                      'g6pd_deficiency',       'risk of haemolysis in G6PD deficiency'),
        ('warfarin',                  'warfarin',              'significant drug interaction with warfarin'),
        ('coumadin',                  'warfarin',              'significant drug interaction with warfarin'),
        ('pacemaker',                 'pacemaker',             'laser therapy — confirm pacemaker compatibility'),
        ('diabetes',                  'diabetes',              'systemic corticosteroids may worsen glycaemic control'),
        ('osteoporosis',              'osteoporosis',          'systemic corticosteroids increase fracture risk'),
        ('vte',                       'vte_history',           'increased thromboembolism risk'),
        ('malignancy',                'malignancy_history',    'caution in patients with prior malignancy'),
        ('active infection',          'active_infection',      'treat active infection before starting immunotherapy'),
        ('active tb',                 'active_tb',             'treat active TB before anti-TNF/biologic therapy'),
        ('inflammatory bowel',        'inflammatory_bowel_disease', 'IL-17 inhibitors may worsen IBD'),
        ('crohn',                     'inflammatory_bowel_disease', 'IL-17 inhibitors contraindicated in active Crohn\'s'),
    ]

    for ci in option.contraindications:
        ci_lower = ci.lower()
        for ci_keyword, ph_key, warning_msg in CI_CHECKS:
            if ci_keyword in ci_lower and ph.get(ph_key):
                warnings.append(f'{option.name}: {warning_msg}')
                break

    return warnings


# ── MAIN INTERFACE ────────────────────────────────────────────────────────────

def recommend_treatment(
    predicted_class: str,
    risk_level: str = 'Moderate',
    patient_history: Optional[dict] = None,
) -> DrugRecommendation:
    """
    Rule-based treatment recommendation.

    Parameters
    ----------
    predicted_class  : HC / IT Fusion output string (e.g. 'Melanoma', 'BCC',
                       'psoriasis', 'eczema', 'tinea')
    risk_level       : From ABCDE risk scoring ('Low' | 'Moderate' | 'High')
    patient_history  : Dict of boolean patient flags for contraindication check
                       (see _check_contraindications for full key list)

    Returns
    -------
    DrugRecommendation with filtered options and any CI warnings
    """
    condition_key = _normalise_condition(predicted_class)
    kb = TREATMENT_KB.get(condition_key, TREATMENT_KB['default'])

    filtered_options = []
    ci_warnings = []
    for opt in kb['options']:
        cw = _check_contraindications(opt, patient_history)
        if cw:
            ci_warnings.extend(cw)
        filtered_options.append(opt)

    notes = list(kb['notes'])
    if ci_warnings:
        notes.append('⚠ Contraindication warnings: ' + '; '.join(ci_warnings))

    return DrugRecommendation(
        condition=predicted_class,
        risk_level=risk_level,
        options=filtered_options,
        surgical_referral=kb['surgical_referral'],
        oncology_referral=kb['oncology_referral'],
        general_notes=notes,
    )


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cases = [
        ('Melanoma',              'High',     {'autoimmune': False, 'braf_wildtype': True}),
        ('BCC',                   'Moderate', {'pregnant': True}),
        ('psoriasis',             'Moderate', {'renal_impairment': True}),
        ('eczema',                'Moderate', {}),
        ('allergic contact dermatitis', 'Low', {}),
        ('tinea',                 'Low',      {'hepatic_impairment': True}),
        ('acne',                  'Moderate', {'pregnant': True}),
        ('herpes zoster',         'Moderate', {'renal_impairment': True}),
        ('cellulitis',            'High',     {'penicillin_allergy': True}),
        ('alopecia areata',       'Moderate', {'vte_history': True}),
        ('vitiligo',              'Low',      {}),
        ('Vasc',                  'Low',      {'asthma': True}),
        ('Nevus',                 'Low',      {}),
        ('AKIEC',                 'Moderate', {}),
        ('SEK',                   'Low',      {}),
        ('Unknown condition xyz', 'Low',      {}),
    ]

    for cls, risk, hist in cases:
        rec = recommend_treatment(cls, risk, hist)
        print(f'\n{"="*65}')
        print(f'  {rec.condition:<30} | Risk: {rec.risk_level}')
        print(f'  Surgical: {rec.surgical_referral} | Oncology: {rec.oncology_referral}')
        print('  Options:')
        for opt in rec.options[:3]:
            print(f'    [{opt.line.upper():8s}] {opt.name:<40} ({opt.route})')
        for note in rec.general_notes:
            if '⚠' in note:
                print(f'  {note}')