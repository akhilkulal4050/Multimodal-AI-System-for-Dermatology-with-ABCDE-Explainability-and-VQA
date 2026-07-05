"""
feature_alignment.py
────────────────────
Feature alignment utilities for IT Fusion.

Handles:
  1. L2 normalisation (for cosine similarity / contrastive learning)
  2. Dimension alignment — pads or projects features to a shared dim
  3. Contrastive alignment loss (InfoNCE / NT-Xent) for image-text pre-training
  4. Feature quality checks and diagnostic statistics

Used during:
  - dataset.ipynb  : to align HC and MedProc feature spaces before saving
  - train_it_fusion.ipynb : as the contrastive pre-training objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# ── 1. NORMALISATION ──────────────────────────────────────────────────────────

def align_features(
    image_features: torch.Tensor,
    text_features : torch.Tensor,
    normalize     : bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align image and text features into the same comparable space.

    Applies L2 normalisation so that cosine similarity == dot product.
    Both feature tensors MUST already have the same dimension.

    Parameters
    ----------
    image_features : (B, D)
    text_features  : (B, D)
    normalize      : whether to L2-normalise (default True)

    Returns
    -------
    (image_features_aligned, text_features_aligned) — both (B, D)
    """
    if normalize:
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features,  dim=-1)
    return image_features, text_features


# ── 2. DIMENSION PROJECTOR ────────────────────────────────────────────────────

class DimAligner(nn.Module):
    """
    Projects features from different source dimensions to a shared target dim.

    Useful when HC produces 512-dim and MedProc produces 768-dim features —
    both need to be projected to the same space for fusion or contrastive loss.
    """

    def __init__(self, image_dim: int, text_dim: int, out_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        ) if image_dim != out_dim else nn.Identity()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        ) if text_dim != out_dim else nn.Identity()

    def forward(
        self,
        image_features: torch.Tensor,
        text_features : torch.Tensor,
        normalize     : bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns L2-normalised projected features, both (B, out_dim).
        """
        img = self.image_proj(image_features)
        txt = self.text_proj(text_features)
        if normalize:
            img = F.normalize(img, dim=-1)
            txt = F.normalize(txt, dim=-1)
        return img, txt


# ── 3. CONTRASTIVE ALIGNMENT LOSS ─────────────────────────────────────────────

class ContrastiveAlignmentLoss(nn.Module):
    """
    InfoNCE / NT-Xent contrastive loss for image-text alignment.

    For each sample in the batch:
      - The (image_i, text_i) pair is the positive pair
      - All (image_i, text_j) where i≠j are negatives

    This is the standard CLIP-style loss used for visual-language pre-training.

    temperature : lower → sharper distribution → harder negatives
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        # Learnable temperature (optional, set learn_temp=True)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(
        self,
        image_features: torch.Tensor,   # (B, D) — must be L2-normalised
        text_features : torch.Tensor,   # (B, D) — must be L2-normalised
        labels        : Optional[torch.Tensor] = None,  # (B,) class labels for supervised
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute bidirectional contrastive loss.

        Parameters
        ----------
        image_features : L2-normalised (B, D)
        text_features  : L2-normalised (B, D)
        labels         : optional class labels for supervised contrastive variant

        Returns
        -------
        loss   : scalar tensor
        metrics: dict with 'i2t_loss', 't2i_loss', 'avg_i2t_acc', 'avg_t2i_acc'
        """
        temp = torch.exp(self.log_temp).clamp(min=0.01, max=1.0)
        B = image_features.size(0)
        device = image_features.device

        # (B, B) cosine similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / temp

        if labels is None:
            # Standard NT-Xent: diagonal is positive
            targets = torch.arange(B, device=device)
        else:
            # Supervised: same-class pairs are all positives
            # (uses soft labels / multi-positive formulation)
            targets = torch.arange(B, device=device)  # fallback to standard

        # Image → Text direction
        i2t_loss = F.cross_entropy(logits, targets)
        # Text → Image direction
        t2i_loss = F.cross_entropy(logits.t(), targets)

        loss = (i2t_loss + t2i_loss) / 2.0

        # Retrieval accuracy
        with torch.no_grad():
            i2t_acc = (logits.argmax(dim=1) == targets).float().mean().item()
            t2i_acc = (logits.t().argmax(dim=1) == targets).float().mean().item()

        return loss, {
            'i2t_loss'    : i2t_loss.item(),
            't2i_loss'    : t2i_loss.item(),
            'i2t_top1_acc': i2t_acc,
            't2i_top1_acc': t2i_acc,
            'temperature' : temp.item(),
        }


# ── 4. COSINE SIMILARITY MATRIX ───────────────────────────────────────────────

def cosine_similarity_matrix(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between all pairs (a_i, b_j).

    a : (N, D)
    b : (M, D)
    Returns : (N, M)
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return torch.matmul(a, b.t())


# ── 5. FEATURE STATISTICS & QUALITY CHECK ─────────────────────────────────────

def feature_quality_report(
    image_features: torch.Tensor,
    text_features : torch.Tensor,
    name          : str = 'batch',
) -> dict:
    """
    Compute diagnostic statistics for a feature batch.
    Useful for detecting collapsed features, dead neurons, or alignment issues.

    Returns a dict with per-modality stats and cross-modal alignment score.
    """

    def _stats(x: torch.Tensor, label: str) -> dict:
        norms = x.norm(dim=-1)
        return {
            f'{label}_mean_norm'  : norms.mean().item(),
            f'{label}_std_norm'   : norms.std().item(),
            f'{label}_feat_std'   : x.std(dim=0).mean().item(),  # feature diversity
            f'{label}_dead_ratio' : (x.abs().max(dim=-1).values < 1e-4).float().mean().item(),
        }

    img_stats = _stats(image_features, 'image')
    txt_stats = _stats(text_features,  'text')

    # Cross-modal alignment: mean diagonal similarity
    img_n = F.normalize(image_features, dim=-1)
    txt_n = F.normalize(text_features,  dim=-1)
    diag_sim = (img_n * txt_n).sum(dim=-1).mean().item()  # mean positive pair similarity

    report = {
        'name'              : name,
        'batch_size'        : image_features.size(0),
        'feature_dim'       : image_features.size(1),
        'cross_modal_sim'   : round(diag_sim, 4),
        **img_stats,
        **txt_stats,
    }

    # Warnings
    warnings = []
    if img_stats['image_feat_std'] < 0.05:
        warnings.append('image features may be collapsed (low diversity)')
    if txt_stats['text_feat_std'] < 0.05:
        warnings.append('text features may be collapsed (low diversity)')
    if diag_sim < 0.1:
        warnings.append('cross-modal alignment is poor — consider contrastive training')
    if img_stats['image_dead_ratio'] > 0.1:
        warnings.append(f'{img_stats["image_dead_ratio"]:.0%} of image samples have near-zero features')

    report['warnings'] = warnings

    return report


# ── 6. NUMPY VERSIONS (for dataset.ipynb pre-processing) ─────────────────────

def numpy_l2_normalize(features: np.ndarray) -> np.ndarray:
    """L2-normalise a numpy feature array row-wise."""
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / (norms + 1e-8)


def numpy_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity for numpy arrays.
    a: (N, D), b: (M, D) → (N, M)
    """
    a = numpy_l2_normalize(a)
    b = numpy_l2_normalize(b)
    return a @ b.T


# ── DEMO ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    B, D = 8, 512

    img_feat = torch.randn(B, D)
    txt_feat = torch.randn(B, D)

    # ── Basic alignment
    img_a, txt_a = align_features(img_feat, txt_feat)
    print(f'Aligned norms — img: {img_a.norm(dim=-1).mean():.4f}, '
          f'txt: {txt_a.norm(dim=-1).mean():.4f}  (should both be 1.0)')

    # ── Contrastive loss
    loss_fn = ContrastiveAlignmentLoss(temperature=0.07)
    loss, metrics = loss_fn(img_a, txt_a)
    print(f'\nContrastive loss: {loss.item():.4f}')
    print(f'i2t Accuracy: {metrics["i2t_top1_acc"]:.1%}')
    print(f't2i Accuracy: {metrics["t2i_top1_acc"]:.1%}')
    print(f'Temperature : {metrics["temperature"]:.4f}')

    # ── DimAligner (different dims)
    aligner = DimAligner(image_dim=1280, text_dim=768, out_dim=512)
    img_large = torch.randn(B, 1280)
    txt_large = torch.randn(B, 768)
    img_proj, txt_proj = aligner(img_large, txt_large)
    print(f'\nDimAligner output: img {img_proj.shape}, txt {txt_proj.shape}')

    # ── Quality report
    report = feature_quality_report(img_a, txt_a, name='test_batch')
    print(f'\nFeature quality report:')
    for k, v in report.items():
        if k != 'warnings':
            print(f'  {k}: {v}')
    print(f'  warnings: {report["warnings"]}')
