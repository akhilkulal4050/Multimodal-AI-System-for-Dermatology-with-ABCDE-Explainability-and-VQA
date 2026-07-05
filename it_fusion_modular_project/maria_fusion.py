"""
maria_fusion.py
───────────────
Standalone MARIA-style cross-modal fusion module for IT Fusion.

This version is a compact, standalone implementation that can be used
when the full fusion.py is not needed (e.g. for testing, ablation, or
lightweight deployment).

It implements:
  1. MARIAFusion — masked multi-head attention fusion of two or more modalities
  2. ModalityProjector — aligns heterogeneous feature dims to a common d_model
  3. FusionClassifier — full pipeline: project → fuse → classify

Architecture matches MARIA paper (Caruso et al., arXiv:2412.14810v2)
but simplified for the 2-modality (image + text) use case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# 1.  MODALITY PROJECTOR
#     Projects variable-dim features to a shared embedding space
# ═════════════════════════════════════════════════════════════════════════════

class ModalityProjector(nn.Module):
    """
    Projects a modality feature vector to a common d_model dimension.

    Input  : (B, in_dim)
    Output : (B, d_model)
    """

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  MASKED CROSS-MODAL ATTENTION
#     Fuses two projected modality tokens with optional missing-token masking
# ═════════════════════════════════════════════════════════════════════════════

class CrossModalAttention(nn.Module):
    """
    Cross-attention block that lets one modality attend to the other.

    When a modality is missing (mask=0), the attention is zeroed out
    and the residual path is used instead (graceful degradation).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_h     = d_model // n_heads
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,           # (B, d_model)
        key_value: torch.Tensor,       # (B, d_model)
        kv_mask: Optional[torch.Tensor] = None,  # (B,) float: 1=present, 0=missing
    ) -> torch.Tensor:
        """Returns: attended query (B, d_model)."""
        B = query.size(0)

        q = self.Q(query).view(B, 1, self.n_heads, self.d_h).transpose(1, 2)
        k = self.K(key_value).view(B, 1, self.n_heads, self.d_h).transpose(1, 2)
        v = self.V(key_value).view(B, 1, self.n_heads, self.d_h).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_h ** 0.5)
        # (B, n_heads, 1, 1)

        if kv_mask is not None:
            # Add -inf where key is missing to prevent attention
            additive = (1.0 - kv_mask.float()) * (-1e9)
            scores = scores + additive.view(B, 1, 1, 1)

        attn = F.relu(F.softmax(scores, dim=-1))  # MARIA paper uses ReLU(softmax)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                   # (B, n_heads, 1, d_h)
        out = out.transpose(1, 2).contiguous().view(B, self.d_model)

        # If KV was masked (missing), zero the cross-attention contribution
        if kv_mask is not None:
            out = out * kv_mask.unsqueeze(-1).float()

        return self.norm(query + self.out(out))


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MARIA FUSION MODULE (standalone, 2-modality version)
# ═════════════════════════════════════════════════════════════════════════════

class MARIAFusion(nn.Module):
    """
    MARIA-style cross-modal fusion for image + text features.

    This is the standalone/minimal version used for:
      - Lightweight inference
      - Ablation studies
      - Import from other modules without loading the full fusion.py

    Supports graceful degradation when one modality is absent.

    Input shapes:
      image_features : (B, image_dim)   — e.g. 512 from HC / EfficientNet
      text_features  : (B, text_dim)    — e.g. 512 from MedProc / BioBERT

    Output:
      fused          : (B, d_model)     — joint representation
    """

    def __init__(
        self,
        image_dim  : int = 512,
        text_dim   : int = 512,
        d_model    : int = 256,
        n_heads    : int = 4,
        n_layers   : int = 2,
        dropout    : float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Modality projectors
        self.image_proj = ModalityProjector(image_dim, d_model, dropout)
        self.text_proj  = ModalityProjector(text_dim,  d_model, dropout)

        # Bidirectional cross-attention layers
        self.img_attends_text = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.txt_attends_img  = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Self-attention for within-modality refinement
        self.image_self_norm = nn.LayerNorm(d_model)
        self.text_self_norm  = nn.LayerNorm(d_model)

        # Final fusion gate: learns how much weight to give each modality
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 2),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features : torch.Tensor,
        image_mask    : Optional[torch.Tensor] = None,  # (B,) 1=present
        text_mask     : Optional[torch.Tensor] = None,  # (B,) 1=present
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        image_features : (B, image_dim)
        text_features  : (B, text_dim)
        image_mask     : (B,) float — 1 if image is available, 0 if missing
        text_mask      : (B,) float — 1 if text is available, 0 if missing

        Returns
        -------
        fused      : (B, d_model) — joint fused embedding
        gate_weights : (B, 2) — [image_weight, text_weight] for interpretability
        """
        B = image_features.size(0)

        if image_mask is None:
            image_mask = torch.ones(B, device=image_features.device)
        if text_mask is None:
            text_mask = torch.ones(B, device=text_features.device)

        # Project to common space
        img_emb = self.image_proj(image_features)  # (B, d_model)
        txt_emb = self.text_proj(text_features)    # (B, d_model)

        # Bidirectional cross-attention
        for img_attn, txt_attn in zip(self.img_attends_text, self.txt_attends_img):
            img_emb = img_attn(img_emb, txt_emb, kv_mask=text_mask)
            txt_emb = txt_attn(txt_emb, img_emb, kv_mask=image_mask)

        # Modality-level masking (zero out missing modalities)
        img_emb = img_emb * image_mask.unsqueeze(-1).float()
        txt_emb = txt_emb * text_mask.unsqueeze(-1).float()

        # Gated fusion
        concat = torch.cat([img_emb, txt_emb], dim=-1)  # (B, 2*d_model)
        gate_weights = self.gate(concat)                  # (B, 2)

        # Weighted combination
        weighted = (
            gate_weights[:, 0:1] * img_emb +
            gate_weights[:, 1:2] * txt_emb
        )                                                 # (B, d_model)

        # Final projection with residual
        fused = self.final_proj(torch.cat([weighted, concat[:, :self.d_model]], dim=-1))

        return fused, gate_weights


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MULTI-MODALITY FUSION (extends to N modalities)
# ═════════════════════════════════════════════════════════════════════════════

class MultiModalFusion(nn.Module):
    """
    Extends MARIAFusion to N modalities by pairwise cross-attention then pooling.
    Used when ABCDE, demographic, and history modalities are also available.

    Input: dict of {name: (B, dim)} tensors + optional masks
    Output: (B, d_model) fused representation
    """

    def __init__(
        self,
        modality_dims : Dict[str, int],
        d_model       : int = 256,
        n_heads       : int = 4,
        n_layers      : int = 2,
        dropout       : float = 0.1,
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.d_model = d_model

        # Per-modality projectors
        self.projectors = nn.ModuleDict({
            name: ModalityProjector(dim, d_model, dropout)
            for name, dim in modality_dims.items()
        })

        # Single shared cross-attention encoder
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn' : nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'ff'   : nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                ),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])

        self.pool_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        modalities : Dict[str, torch.Tensor],
        masks      : Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        modalities : {name: (B, dim)} for each available modality
        masks      : {name: (B,)} 1=present, 0=missing (optional)

        Returns
        -------
        (B, d_model) fused representation
        """
        if masks is None:
            masks = {}

        B = next(iter(modalities.values())).size(0)
        device = next(iter(modalities.values())).device

        # Project and stack modalities into sequence
        tokens = []
        token_mask = []
        for name in self.modality_names:
            if name not in modalities:
                # Missing modality — insert zero token with mask=0
                tokens.append(torch.zeros(B, 1, self.d_model, device=device))
                token_mask.append(torch.zeros(B, 1, dtype=torch.bool, device=device))
            else:
                feat = modalities[name]
                proj = self.projectors[name](feat).unsqueeze(1)  # (B, 1, d_model)
                tokens.append(proj)
                m = masks.get(name, torch.ones(B, device=device))
                token_mask.append((m == 0).unsqueeze(1))  # True=ignore in attn

        seq = torch.cat(tokens, dim=1)             # (B, n_mod, d_model)
        key_pad_mask = torch.cat(token_mask, dim=1)  # (B, n_mod) True=ignore

        # Transformer self-attention over the modality tokens
        h = seq
        for layer in self.layers:
            attn_out, _ = layer['attn'](
                h, h, h,
                key_padding_mask=key_pad_mask
            )
            h = layer['norm1'](h + attn_out)
            h = layer['norm2'](h + layer['ff'](h))

        # Mean pool over present modalities
        present = (~key_pad_mask).float().unsqueeze(-1)  # (B, n_mod, 1)
        pooled = (h * present).sum(dim=1) / (present.sum(dim=1) + 1e-6)  # (B, d_model)

        return self.pool_proj(pooled)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  FUSION CLASSIFIER  (project → fuse → classify)
# ═════════════════════════════════════════════════════════════════════════════

class FusionClassifier(nn.Module):
    """
    End-to-end: fuse image + text features, then classify.

    Used by train_it_fusion.ipynb for training and inference.
    """

    def __init__(
        self,
        image_dim  : int = 512,
        text_dim   : int = 512,
        d_model    : int = 256,
        n_classes  : int = 7,
        n_heads    : int = 4,
        n_layers   : int = 2,
        dropout    : float = 0.1,
    ):
        super().__init__()

        self.fusion = MARIAFusion(
            image_dim=image_dim, text_dim=text_dim,
            d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features : torch.Tensor,
        image_mask    : Optional[torch.Tensor] = None,
        text_mask     : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits       : (B, n_classes)
        fused        : (B, d_model)
        gate_weights : (B, 2)  — image vs text contribution
        """
        fused, gate_weights = self.fusion(
            image_features, text_features, image_mask, text_mask
        )
        logits = self.classifier(fused)
        return logits, fused, gate_weights


# ═════════════════════════════════════════════════════════════════════════════
# 6.  DEMO
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    B = 4

    # Simulate HC image features + MedProc text features
    img_feat = torch.randn(B, 512)
    txt_feat = torch.randn(B, 512)

    # One sample has missing text
    txt_mask  = torch.tensor([1., 1., 0., 1.])
    img_mask  = torch.ones(B)

    # ── 2-modality MARIAFusion ────────────────────────────────────────────────
    fusion = MARIAFusion(image_dim=512, text_dim=512, d_model=256)
    fused, gates = fusion(img_feat, txt_feat, img_mask, txt_mask)
    print(f'MARIAFusion output: {fused.shape}   gates: {gates.shape}')
    print(f'  Sample 0 gate weights (img/txt): {gates[0].tolist()}')
    print(f'  Sample 2 gate weights (txt missing): {gates[2].tolist()}')

    # ── FusionClassifier ──────────────────────────────────────────────────────
    classifier = FusionClassifier(image_dim=512, text_dim=512,
                                  d_model=256, n_classes=7)
    logits, fused2, gates2 = classifier(img_feat, txt_feat, img_mask, txt_mask)
    probs = torch.softmax(logits, dim=-1)
    print(f'\nFusionClassifier logits: {logits.shape}')
    print(f'  Sample 0 probs: {[round(p,3) for p in probs[0].tolist()]}')

    # ── MultiModalFusion (5 modalities) ──────────────────────────────────────
    multi = MultiModalFusion(
        modality_dims={
            'image'      : 512,
            'text'       : 512,
            'abcde'      : 13,
            'demographics': 3,
        },
        d_model=128,
    )
    multi_out = multi(
        modalities={
            'image'      : img_feat,
            'text'       : txt_feat,
            'abcde'      : torch.randn(B, 13),
            'demographics': torch.randn(B, 3),
        },
        masks={
            'image'      : img_mask,
            'text'       : txt_mask,
            'abcde'      : torch.ones(B),
            'demographics': torch.tensor([1., 1., 0., 0.]),
        }
    )
    print(f'\nMultiModalFusion (5 mod) output: {multi_out.shape}')
