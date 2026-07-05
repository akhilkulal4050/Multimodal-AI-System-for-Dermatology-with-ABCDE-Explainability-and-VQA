"""
abcde_computation.py
────────────────────
Computes the ABCDE criteria scores for a single skin lesion.

Inputs (all produced by upstream stages):
  • Binary segmentation mask     → from ARCUNet (Stage 2)
  • ROI cropped image (224×224)  → from SLRC     (Stage 3)
  • Classification features      → from HC        (Stage 4)
  • Evolution score + text       → from MedProc   (Stage 5)

Output:
  • Dict with keys A, B, C, D, E (each 0.0–1.0) + risk_score + evidence list

ABCDE interpretation reference:
  A – Asymmetry   : 0 = perfectly symmetric,  1 = highly asymmetric
  B – Border      : 0 = smooth regular border, 1 = highly irregular
  C – Color       : 0 = uniform single color,  1 = multicolour / high variance
  D – Diameter    : 0 = < 2mm,                 1 = > 10mm (clinically significant)
  E – Evolution   : 0 = no change reported,    1 = significant recent change

Risk score = weighted sum (weights tuned to clinical literature):
  risk = 0.25*A + 0.20*B + 0.20*C + 0.15*D + 0.20*E
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class ABCDEResult:
    A: float           # Asymmetry  [0, 1]
    B: float           # Border     [0, 1]
    C: float           # Color      [0, 1]
    D: float           # Diameter   [0, 1]
    E: float           # Evolution  [0, 1]
    risk_score: float  # Weighted composite [0, 1]
    risk_level: str    # 'Low' | 'Moderate' | 'High'
    evidence: List[str]
    raw_metrics: dict  # Detailed intermediate values for explainability

    def to_dict(self):
        return asdict(self)


# ── WEIGHTS (tunable) ─────────────────────────────────────────────────────────
ABCDE_WEIGHTS = {
    'A': 0.25,
    'B': 0.20,
    'C': 0.20,
    'D': 0.15,
    'E': 0.20,
}

# Clinical risk thresholds
RISK_THRESHOLDS = {'High': 0.65, 'Moderate': 0.40}


# ── A – ASYMMETRY ─────────────────────────────────────────────────────────────

def compute_asymmetry(mask: np.ndarray) -> Tuple[float, dict]:
    """
    Asymmetry is computed by comparing the lesion halves after aligning
    to the principal axis (PCA-based rotation).

    Strategy:
      1. Find the principal axis via image moments.
      2. Rotate mask so the major axis is horizontal.
      3. Flip horizontally and vertically, compute overlap ratios.
      4. Score = 1 − (min_overlap / lesion_area)

    Returns: (score 0–1, raw_metrics dict)
    """
    raw = {}
    if mask is None or mask.sum() == 0:
        return 0.5, {'note': 'empty mask — defaulting to 0.5'}

    # Binary mask
    m = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
    lesion_area = float(m.sum())
    raw['lesion_area_px'] = lesion_area

    # Image moments → rotation angle
    moments = cv2.moments(m)
    if moments['mu20'] == moments['mu02'] == 0:
        return 0.5, {'note': 'degenerate moments'}

    angle = 0.5 * np.arctan2(2 * moments['mu11'],
                              moments['mu20'] - moments['mu02'])
    cx = int(moments['m10'] / (moments['m00'] + 1e-6))
    cy = int(moments['m01'] / (moments['m00'] + 1e-6))
    raw['centroid'] = (cx, cy)
    raw['angle_rad'] = float(angle)

    # Rotate mask to align principal axis
    h, w = m.shape
    rot_mat = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1.0)
    rotated = cv2.warpAffine(m, rot_mat, (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Horizontal asymmetry (fold left / right)
    left  = rotated[:, :w//2]
    right = np.fliplr(rotated[:, w//2:])
    min_w = min(left.shape[1], right.shape[1])
    h_overlap = np.logical_and(left[:, :min_w], right[:, :min_w]).sum()
    h_union    = np.logical_or(left[:, :min_w], right[:, :min_w]).sum()
    h_iou = h_overlap / (h_union + 1e-6)

    # Vertical asymmetry (fold top / bottom)
    top    = rotated[:h//2, :]
    bottom = np.flipud(rotated[h//2:, :])
    min_h  = min(top.shape[0], bottom.shape[0])
    v_overlap = np.logical_and(top[:min_h], bottom[:min_h]).sum()
    v_union    = np.logical_or(top[:min_h], bottom[:min_h]).sum()
    v_iou = v_overlap / (v_union + 1e-6)

    # Asymmetry score: low IoU → high asymmetry
    score = 1.0 - (h_iou + v_iou) / 2.0
    score = float(np.clip(score, 0.0, 1.0))

    raw['h_iou'] = float(h_iou)
    raw['v_iou'] = float(v_iou)
    return score, raw


# ── B – BORDER IRREGULARITY ───────────────────────────────────────────────────

def compute_border(mask: np.ndarray) -> Tuple[float, dict]:
    """
    Border score based on:
      1. Compactness index: C = perimeter² / (4π × area)
                            Perfect circle → 1.0; irregular → higher
      2. Fractal dimension approximation via box-counting.
      3. Normalised score mapped to [0, 1].

    Returns: (score 0–1, raw_metrics dict)
    """
    raw = {}
    if mask is None or mask.sum() == 0:
        return 0.5, {'note': 'empty mask'}

    m = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.5, {'note': 'no contours found'}

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)

    if area < 1:
        return 0.5, {'note': 'area too small'}

    # Compactness: circle = 1, irregular > 1
    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
    raw['compactness'] = float(compactness)

    # Convexity ratio: how much of convex hull is filled
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / (hull_area + 1e-6)
    raw['convexity_ratio'] = float(convexity)

    # Notching / irregularity via contour curvature variance
    if len(cnt) > 5:
        ellipse = cv2.fitEllipse(cnt)
        (_, _), (MA, ma), _ = ellipse
        ellipse_ratio = min(MA, ma) / (max(MA, ma) + 1e-6)
        raw['ellipse_aspect_ratio'] = float(ellipse_ratio)
    else:
        ellipse_ratio = 1.0

    # Combine into border score [0, 1]
    # compactness ≈ 1 → smooth, > 2 → irregular
    compactness_score = np.clip((compactness - 1.0) / 3.0, 0.0, 1.0)
    convexity_score   = 1.0 - float(convexity)  # low convexity → irregular
    score = 0.6 * compactness_score + 0.4 * convexity_score
    score = float(np.clip(score, 0.0, 1.0))

    raw['compactness_score'] = float(compactness_score)
    raw['convexity_score']   = float(convexity_score)
    return score, raw


# ── C – COLOR VARIATION ───────────────────────────────────────────────────────

def compute_color(roi_img: np.ndarray, mask: np.ndarray) -> Tuple[float, dict]:
    """
    Color score based on:
      1. Number of dominant color clusters within the lesion (k-means, k=5).
      2. Inter-cluster color distance (ΔE in Lab space).
      3. Standard deviation of each Lab channel within lesion.

    Returns: (score 0–1, raw_metrics dict)
    """
    raw = {}
    if roi_img is None or mask is None:
        return 0.5, {'note': 'missing image or mask'}

    m = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
    if m.sum() < 50:
        return 0.5, {'note': 'lesion too small for colour analysis'}

    # Extract lesion pixels
    if roi_img.ndim == 2:
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)

    # Normalise to BGR (OpenCV convention) regardless of source
    # PIL / torchvision / SLRC output RGB; cv2.imread outputs BGR.
    # We detect likely RGB by checking if the image came as a 3-channel uint8
    # and convert to BGR so COLOR_BGR2LAB gives correct Lab values.
    # Caller can pass channel_order='bgr' to skip conversion.
    # Default assumption: input is RGB (safer for the SLRC→PIL pipeline).
    roi_bgr = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    lesion_pixels = img_lab[m == 1]   # shape (N, 3)
    raw['n_lesion_pixels'] = int(len(lesion_pixels))

    # Channel std dev
    std_L, std_a, std_b = lesion_pixels.std(axis=0)
    raw['std_L'] = float(std_L)
    raw['std_a'] = float(std_a)
    raw['std_b'] = float(std_b)

    # Colour std score: normalise by empirical max (benign ≈ 15, malignant ≈ 40+)
    std_composite = (std_L + std_a + std_b) / 3.0
    std_score = float(np.clip(std_composite / 35.0, 0.0, 1.0))
    raw['std_composite'] = float(std_composite)

    # k-means dominant colours
    try:
        from sklearn.cluster import MiniBatchKMeans
        K = min(5, len(lesion_pixels) // 10 + 1)
        km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42)
        km.fit(lesion_pixels)
        centers = km.cluster_centers_  # (K, 3)
        labels = km.labels_
        counts = np.bincount(labels)
        active_clusters = (counts > len(lesion_pixels) * 0.05).sum()  # > 5% share
        raw['active_color_clusters'] = int(active_clusters)

        # Mean inter-cluster ΔE
        from itertools import combinations
        diffs = [np.linalg.norm(centers[i] - centers[j])
                 for i, j in combinations(range(len(centers)), 2)]
        mean_delta_e = np.mean(diffs) if diffs else 0.0
        raw['mean_delta_e'] = float(mean_delta_e)
        cluster_score = float(np.clip(mean_delta_e / 60.0, 0.0, 1.0))
    except Exception:
        active_clusters = 1
        cluster_score   = 0.0
        raw['cluster_error'] = True

    score = 0.5 * std_score + 0.5 * cluster_score
    score = float(np.clip(score, 0.0, 1.0))
    raw['std_score']     = std_score
    raw['cluster_score'] = cluster_score
    return score, raw


# ── D – DIAMETER ──────────────────────────────────────────────────────────────

def compute_diameter(mask: np.ndarray,
                     pixel_spacing_mm: float = 0.1) -> Tuple[float, dict]:
    """
    Diameter score from mask bounding box and equivalent circle diameter.
    pixel_spacing_mm: physical size of one pixel in mm (default 0.1 mm/px
    for smartphone dermoscopy; adjust for clinical equipment).

    Clinical threshold: > 6mm is a key melanoma criterion.

    Returns: (score 0–1, raw_metrics dict)
    """
    raw = {}
    if mask is None or mask.sum() == 0:
        return 0.5, {'note': 'empty mask'}

    m = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)

    # Bounding box diameter
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return 0.5, {'note': 'no lesion pixels'}

    bbox_w_px = float(xs.max() - xs.min() + 1)
    bbox_h_px = float(ys.max() - ys.min() + 1)
    max_diameter_px = max(bbox_w_px, bbox_h_px)

    # Equivalent circle diameter from area
    area_px   = float(m.sum())
    eq_diam_px = 2.0 * np.sqrt(area_px / np.pi)

    # Convert to mm
    max_diam_mm = max_diameter_px * pixel_spacing_mm
    eq_diam_mm  = eq_diam_px * pixel_spacing_mm

    raw['max_diameter_mm'] = float(max_diam_mm)
    raw['eq_diameter_mm']  = float(eq_diam_mm)
    raw['pixel_spacing_mm']= pixel_spacing_mm

    # Sigmoid mapping centred at 6mm (clinical threshold)
    # score ≈ 0 for < 2mm, ≈ 0.5 at 6mm, ≈ 1 for > 15mm
    score = 1.0 / (1.0 + np.exp(-0.5 * (max_diam_mm - 6.0)))
    score = float(np.clip(score, 0.0, 1.0))
    return score, raw


# ── E – EVOLUTION ─────────────────────────────────────────────────────────────

def compute_evolution(evolution_score: Optional[float],
                      symptom_keywords: Optional[List[str]] = None) -> Tuple[float, dict]:
    """
    Evolution score is provided directly by MedProc (Stage 5).
    This function validates, normalises, and enriches it with keyword boosting.

    evolution_score: MedProc output in [0, 1] (0 = no change, 1 = rapid change)
    symptom_keywords: list of extracted keywords from MedProc clinical NLP
                      e.g. ['grew', 'colour change', 'bleeding', 'new lesion']

    Returns: (score 0–1, raw_metrics dict)
    """
    raw = {}

    if evolution_score is None:
        base_score = 0.3  # unknown → assume low-moderate (conservative)
        raw['source'] = 'default_no_medproc'
    else:
        base_score = float(np.clip(evolution_score, 0.0, 1.0))
        raw['source'] = 'medproc'
        raw['raw_evolution_score'] = base_score

    # Keyword boosting (each keyword adds a small increment)
    HIGH_CONCERN_KEYWORDS = {
        'grew', 'growth', 'larger', 'spreading', 'increased size',
        'bleed', 'bleeding', 'oozing',
        'colour change', 'color change', 'changed color',
        'new lesion', 'new spot',
        'rapid', 'fast', 'quickly',
        'painful', 'ulcerated', 'ulceration',
    }

    boost = 0.0
    matched = []
    if symptom_keywords:
        for kw in symptom_keywords:
            kw_lower = kw.lower().strip()
            for hkw in HIGH_CONCERN_KEYWORDS:
                if hkw in kw_lower:
                    boost += 0.05
                    matched.append(kw)
                    break
    boost = min(boost, 0.25)  # cap total boost at 0.25
    raw['keyword_boost'] = float(boost)
    raw['matched_keywords'] = matched

    score = float(np.clip(base_score + boost, 0.0, 1.0))
    return score, raw


# ── RISK SCORING ──────────────────────────────────────────────────────────────

def compute_risk(A, B, C, D, E):
    w = ABCDE_WEIGHTS
    risk = w['A']*A + w['B']*B + w['C']*C + w['D']*D + w['E']*E
    if risk >= RISK_THRESHOLDS['High']:
        level = 'High'
    elif risk >= RISK_THRESHOLDS['Moderate']:
        level = 'Moderate'
    else:
        level = 'Low'
    return float(np.clip(risk, 0.0, 1.0)), level


def build_evidence(A, B, C, D, E, raw_A, raw_B, raw_C, raw_D, raw_E):
    """Generate human-readable clinical evidence strings."""
    evidence = []
    if A > 0.5:
        evidence.append(f'Asymmetric lesion (score {A:.2f}) — '
                        f'H-axis IoU {raw_A.get("h_iou", "?"):.2f}')
    if B > 0.5:
        evidence.append(f'Irregular border (score {B:.2f}) — '
                        f'compactness {raw_B.get("compactness", "?"):.2f}')
    if C > 0.5:
        evidence.append(f'Multiple colours (score {C:.2f}) — '
                        f'{raw_C.get("active_color_clusters", "?")} clusters, '
                        f'ΔE {raw_C.get("mean_delta_e", 0):.1f}')
    if D > 0.5:
        evidence.append(f'Large diameter (score {D:.2f}) — '
                        f'{raw_D.get("max_diameter_mm", 0):.1f} mm')
    if E > 0.5:
        evidence.append(f'Evolution reported (score {E:.2f}) — '
                        f'{", ".join(raw_E.get("matched_keywords", [])[:3]) or "MedProc signal"}')
    return evidence


# ── MAIN INTERFACE ────────────────────────────────────────────────────────────

def compute_abcde(
    mask: np.ndarray,
    roi_img: np.ndarray,
    evolution_score: Optional[float] = None,
    symptom_keywords: Optional[List[str]] = None,
    pixel_spacing_mm: float = 0.1,
) -> ABCDEResult:
    """
    Unified ABCDE computation called by the IT Fusion stage.

    Parameters
    ----------
    mask              : Binary mask from ARCUNet (H×W, uint8 or float)
    roi_img           : 224×224 BGR ROI from SLRC
    evolution_score   : Float [0,1] from MedProc Stage 5
    symptom_keywords  : List of NLP-extracted keywords from MedProc
    pixel_spacing_mm  : Physical pixel size in mm (default 0.1)

    Returns
    -------
    ABCDEResult dataclass
    """
    A, raw_A = compute_asymmetry(mask)
    B, raw_B = compute_border(mask)
    C, raw_C = compute_color(roi_img, mask)
    D, raw_D = compute_diameter(mask, pixel_spacing_mm)
    E, raw_E = compute_evolution(evolution_score, symptom_keywords)

    risk_score, risk_level = compute_risk(A, B, C, D, E)
    evidence = build_evidence(A, B, C, D, E, raw_A, raw_B, raw_C, raw_D, raw_E)

    return ABCDEResult(
        A=A, B=B, C=C, D=D, E=E,
        risk_score=risk_score,
        risk_level=risk_level,
        evidence=evidence,
        raw_metrics={
            'A': raw_A, 'B': raw_B, 'C': raw_C,
            'D': raw_D, 'E': raw_E,
        }
    )


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json

    # Create a synthetic circular lesion mask
    mask_test = np.zeros((224, 224), dtype=np.uint8)
    cv2.ellipse(mask_test, (112, 112), (60, 45), 30, 0, 360, 255, -1)
    # Add some irregularity
    cv2.circle(mask_test, (150, 90), 15, 255, -1)

    # Create a synthetic multi-colour ROI
    roi_test = np.zeros((224, 224, 3), dtype=np.uint8)
    roi_test[50:170, 50:170] = [80, 40, 20]   # dark brown
    roi_test[90:130, 80:160] = [30, 60, 90]   # reddish
    roi_test[110:150, 100:140] = [200, 200, 50]  # yellow-white

    result = compute_abcde(
        mask=mask_test,
        roi_img=roi_test,
        evolution_score=0.72,
        symptom_keywords=['grew larger', 'bleeding', 'itching'],
        pixel_spacing_mm=0.15,
    )

    print('=== ABCDE Test Result ===')
    print(f'A (Asymmetry) : {result.A:.3f}')
    print(f'B (Border)    : {result.B:.3f}')
    print(f'C (Color)     : {result.C:.3f}')
    print(f'D (Diameter)  : {result.D:.3f}  [{result.raw_metrics["D"].get("max_diameter_mm",0):.1f} mm]')
    print(f'E (Evolution) : {result.E:.3f}')
    print(f'Risk Score    : {result.risk_score:.3f}  → {result.risk_level}')
    print(f'Evidence      : {result.evidence}')
