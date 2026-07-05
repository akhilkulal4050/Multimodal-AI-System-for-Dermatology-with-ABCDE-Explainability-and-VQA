"""
hc_model.py
───────────
Stage 4 (HC) — Hierarchical Classifier model definition.

Architecture:
  EfficientFormerV2-S2 backbone (pretrained ImageNet)
  → Shared 288-dim features
  → Main head  (Benign / Malignant)
  → Sub  head  (8 skin conditions, conditional on main logits)

This file is the single source of truth for the HC model architecture.
It is used by:
  - HC/HC_improved.ipynb          (training)
  - it_fusion_modular_project/    (feature extraction for MARIA dataset)
"""

import os
import sys
import torch
import torch.nn as nn

# ── EfficientFormerV2 import ────────────────────────────────────────────────
HC_DIR = os.path.join(os.path.dirname(__file__), '..', 'HC')
EFF_DIR = os.path.join(HC_DIR, 'EfficientFormer-main')
sys.path.insert(0, EFF_DIR)

from models.efficientformer_v2 import efficientformerv2_s2, efficientformerv2_s1


# ═════════════════════════════════════════════════════════════════════════════
# SE Block — Squeeze-and-Excitation channel attention
# ═════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C)
        w = self.fc(x.unsqueeze(-1))   # (B, C)
        return x * w


# ═════════════════════════════════════════════════════════════════════════════
# Hierarchical Classifier
# ═════════════════════════════════════════════════════════════════════════════

class HierarchicalClassifier(nn.Module):
    """
    Two-level hierarchical skin lesion classifier.

    Architecture:
      EfficientFormerV2 backbone
      → Shared features (288-dim for S2, 224-dim for S1)
      → Main head  (benign / malignant)
      → Sub  head  (input = features + main_logits ← conditional hierarchy)

    The sub-head receives the main-class logits concatenated with the
    backbone features so it can "see" which broad category is predicted.
    """

    def __init__(self, num_main=2, num_sub=8, model_type='s2', pretrained=True):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────
        if model_type == 's2':
            self.backbone = efficientformerv2_s2(pretrained=pretrained)
            feat_dim = 288
        else:
            self.backbone = efficientformerv2_s1(pretrained=pretrained)
            feat_dim = 224

        # Remove the original classification head
        self.backbone.head = nn.Identity()
        if hasattr(self.backbone, 'dist_head'):
            self.backbone.dist_head = nn.Identity()

        self.feat_dim = feat_dim

        # ── Main head (benign / malignant) ────────────────────────────
        self.main_head = nn.Sequential(
            SEBlock(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_main),
        )

        # ── Sub head (conditional: features + main logits) ────────────
        sub_in = feat_dim + num_main
        self.sub_head = nn.Sequential(
            nn.Linear(sub_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_sub),
        )

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        main_logits = self.main_head(feats)

        # Concatenate detached main logits with features for sub-head
        sub_in     = torch.cat([feats, main_logits.detach()], dim=1)
        sub_logits = self.sub_head(sub_in)

        return main_logits, sub_logits

    def extract_features(self, x):
        """Return 288-dim backbone features only (no classification heads)."""
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        return feats


# ═════════════════════════════════════════════════════════════════════════════
# Helper — load a trained HC checkpoint
# ═════════════════════════════════════════════════════════════════════════════

# Default checkpoint paths
HC_CKPT_V2 = os.path.join(HC_DIR, 'checkpoints', 'hc_best_v2.pt')
HC_CKPT_V1 = os.path.join(HC_DIR, 'checkpoints', 'hc_best.pt')


def load_hc_model(checkpoint_path=None, device='cpu', pretrained_backbone=False):
    """
    Load the full HierarchicalClassifier with trained weights.

    Args:
        checkpoint_path: Path to .pt file. Auto-detected if None.
        device: 'cpu' or 'cuda'.
        pretrained_backbone: Whether to use ImageNet pretrained backbone
                             (only matters if no checkpoint is found).

    Returns:
        model: HierarchicalClassifier in eval mode.
    """
    if checkpoint_path is None:
        if os.path.exists(HC_CKPT_V2):
            checkpoint_path = HC_CKPT_V2
        elif os.path.exists(HC_CKPT_V1):
            checkpoint_path = HC_CKPT_V1

    model = HierarchicalClassifier(pretrained=pretrained_backbone)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        sd = ckpt.get('model_sd', ckpt)
        model.load_state_dict(sd)
        print(f'✓ Loaded HC model from {os.path.basename(checkpoint_path)}')
    else:
        print('⚠ No HC checkpoint found — using uninitialised weights')

    return model.to(device).eval()


def load_hc_backbone(checkpoint_path=None, device='cpu'):
    """
    Load ONLY the EfficientFormerV2-S2 backbone with trained weights.
    Returns a lightweight feature extractor (288-dim output).
    No classification heads are created — saves memory.

    Args:
        checkpoint_path: Path to .pt file. Auto-detected if None.
        device: 'cpu' or 'cuda'.

    Returns:
        backbone: EfficientFormerV2-S2 in eval mode (output: 288-dim).
    """
    if checkpoint_path is None:
        if os.path.exists(HC_CKPT_V2):
            checkpoint_path = HC_CKPT_V2
        elif os.path.exists(HC_CKPT_V1):
            checkpoint_path = HC_CKPT_V1

    backbone = efficientformerv2_s2(pretrained=False)
    backbone.head = nn.Identity()
    if hasattr(backbone, 'dist_head'):
        backbone.dist_head = nn.Identity()

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        sd = ckpt.get('model_sd', ckpt)
        # Extract only backbone.* keys and strip the prefix
        backbone_sd = {k.replace('backbone.', ''): v
                       for k, v in sd.items() if k.startswith('backbone.')}
        backbone.load_state_dict(backbone_sd, strict=False)
        print(f'✓ Loaded HC backbone from {os.path.basename(checkpoint_path)}')
    else:
        print('⚠ No HC checkpoint found — using uninitialised backbone')

    return backbone.to(device).eval()


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== HC Model Module ===')
    model = HierarchicalClassifier(pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  HierarchicalClassifier params: {n_params:,}')
    print(f'  Backbone feat_dim: {model.feat_dim}')

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        main_logits, sub_logits = model(x)
        feats = model.extract_features(x)
    print(f'  main_logits: {main_logits.shape}')
    print(f'  sub_logits:  {sub_logits.shape}')
    print(f'  features:    {feats.shape}')
    print('✓ All checks passed')
