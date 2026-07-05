"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ARCUNet — Attention Residual Convolutional U-Net               ║
║                  Skin Lesion Segmentation  |  v2  |  M.Tech Project         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This file contains ONLY:                                                   ║
║    • Metrics            (dice_coef, jaccard_index, pixel_accuracy)          ║
║    • Loss functions     (DiceLoss, ComboLoss, DeepSupervisionLoss)          ║
║    • Model blocks       (ResidualConv, AttentionGate)                       ║
║    • Model              (ARCUNet)                                           ║
║    • Checkpoint helpers (load_model, save_model)                            ║
║    • Inference helpers  (predict, predict_proba)                            ║
║                                                                             ║
║  Dataset loading, augmentation and training loop → ARCUNet_Train.ipynb     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def dice_coef(pred: torch.Tensor,
              target: torch.Tensor,
              smooth: float = 1.0,
              apply_sigmoid: bool = True) -> torch.Tensor:
    """
    Dice Coefficient  (= F1 score for binary segmentation).

    Formula:
        Dice = (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)

    Args:
        pred          : raw logits  (B, 1, H, W)
        target        : binary mask (B, 1, H, W), values in {0, 1}
        smooth        : Laplace smoothing to avoid division by zero
        apply_sigmoid : set False if pred is already a probability

    Returns:
        Scalar tensor — batch-mean Dice score in [0, 1]
    """
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    pred  = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * inter + smooth) / (denom + smooth)).mean()


def jaccard_index(pred: torch.Tensor,
                  target: torch.Tensor,
                  smooth: float = 1.0,
                  apply_sigmoid: bool = True) -> torch.Tensor:
    """
    Jaccard Index  (= Intersection-over-Union, IoU).

    Formula:
        IoU = (|pred ∩ target| + smooth) / (|pred ∪ target| + smooth)

    Args / Returns: same convention as dice_coef.
    """
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    pred  = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean()


def pixel_accuracy(pred: torch.Tensor,
                   target: torch.Tensor,
                   apply_sigmoid: bool = True) -> torch.Tensor:
    """
    Pixel-wise classification accuracy.

    Returns:
        Fraction of pixels correctly classified — scalar in [0, 1].
    """
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    return ((pred > 0.5).float() == target).float().mean()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Soft (differentiable) Dice Loss.

    Unlike the metric above, this uses *soft probabilities* (not hard 0/1
    predictions) so the gradient is non-zero everywhere — critical for training.

    Loss = 1 - SoftDice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits).view(logits.size(0), -1)   # (B, H*W)
        targets = targets.view(targets.size(0), -1)                # (B, H*W)
        inter   = (probs * targets).sum(dim=1)
        denom   = probs.sum(dim=1) + targets.sum(dim=1)
        dice    = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class ComboLoss(nn.Module):
    """
    Combo Loss  =  alpha * BCE  +  (1 - alpha) * SoftDice

    WHY THIS BEATS PURE BCE:
    ─────────────────────────────────────────────────────────────────────────
    • BCEWithLogitsLoss penalises each pixel equally → biased toward the
      dominant background class on small lesions.
    • DiceLoss directly maximises the Dice overlap score — the primary
      evaluation metric for segmentation tasks.
    • Together they balance pixel-level correctness (BCE) with global
      region overlap (Dice), giving the best of both objectives.
    ─────────────────────────────────────────────────────────────────────────

    Args:
        alpha  : weight on BCE term (0.5 = equal blend)
        smooth : smoothing for Dice denominator
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.bce   = nn.BCEWithLogitsLoss()
        self.dice  = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.alpha * self.bce(logits, targets)
                + (1.0 - self.alpha) * self.dice(logits, targets))


class DeepSupervisionLoss(nn.Module):
    """
    Deep-Supervision-Aware Wrapper around ComboLoss.

    ARCUNet.forward() returns a *tuple* during training:
        (main_logit, ds4_logit, ds3_logit, ds2_logit)

    This loss applies ComboLoss to every output and combines them:
        total = (1 - w_aux) * L(main)
              + (w_aux / 3)  * [L(ds4) + L(ds3) + L(ds2)]

    WHY DEEP SUPERVISION HELPS:
    ─────────────────────────────────────────────────────────────────────────
    • Gradients must travel from the final output all the way back through
      the entire decoder to reach encoder layers — this becomes weak.
    • Deep supervision forces each decoder stage to independently produce
      a valid segmentation, injecting strong gradients at every depth.
    • Result: faster convergence and better feature learning at all scales.
    ─────────────────────────────────────────────────────────────────────────

    Args:
        aux_weight : total weight assigned to the three auxiliary heads.
                     Main output receives (1 - aux_weight).
    """

    def __init__(self, aux_weight: float = 0.4):
        super().__init__()
        self.base_loss  = ComboLoss(alpha=0.5)
        self.aux_weight = aux_weight

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            main     = outputs[0]
            aux_list = outputs[1:]
            loss     = (1.0 - self.aux_weight) * self.base_loss(main, targets)
            w_each   = self.aux_weight / max(len(aux_list), 1)
            for aux in aux_list:
                loss = loss + w_each * self.base_loss(aux, targets)
            return loss
        # Eval mode — outputs is a plain tensor
        return self.base_loss(outputs, targets)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class ResidualConv(nn.Module):
    """
    Residual Convolution Block.

    Structure:
        ┌─ Conv3×3 → BN → ReLU → Dropout2d → Conv3×3 → BN ─┐
        x                                                     ⊕ → ReLU → out
        └──────────────── Conv1×1 → BN ──────────────────────┘

    Improvements over v1:
        • Dropout2d(p) between the two conv layers for regularisation.
          Channel-wise dropout is effective for convolutional networks and
          reduces overfitting on repeated ISIC dermoscopy patterns.
        • bias=False on Conv2d layers followed by BatchNorm (bias is
          redundant and removing it gives cleaner gradients).
        • BatchNorm added to the skip branch for training stability.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    """
    Soft Attention Gate.

    Produces a spatial attention map α ∈ (0,1)^{H×W} that suppresses
    irrelevant background regions in the encoder skip connection before
    it is concatenated with the upsampled decoder feature map.

    Architecture:
        g  (decoder gating signal)  ──→ W_g (1×1 Conv + BN) ──┐
                                                                ⊕ → ReLU → psi → α (Sigmoid)
        x  (encoder skip features)  ──→ W_x (1×1 Conv + BN) ──┘

    Output:  α ⊙ x   (attended skip features)

    Args:
        F_g   : channels in gating signal  (from decoder)
        F_l   : channels in skip signal    (from encoder)
        F_int : bottleneck channels inside the attention block
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g  = nn.Sequential(
            nn.Conv2d(F_g,   F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x  = nn.Sequential(
            nn.Conv2d(F_l,   F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi  = nn.Sequential(
            nn.Conv2d(F_int, 1,     kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ARCUNet  (Full Model)
# ══════════════════════════════════════════════════════════════════════════════

class ARCUNet(nn.Module):
    """
    ARCUNet v2 — Attention Residual Convolutional U-Net with Deep Supervision.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  ENCODER                                                            │
    │  Input (B, 3, H, W)                                                 │
    │  enc1  ResidualConv(3   → 64 )  → e1 (B,  64, H,    W   )         │
    │  pool → enc2  ResidualConv(64  → 128)  → e2 (B, 128, H/2,  W/2 ) │
    │  pool → enc3  ResidualConv(128 → 256)  → e3 (B, 256, H/4,  W/4 ) │
    │  pool → enc4  ResidualConv(256 → 512)  → e4 (B, 512, H/8,  W/8 ) │
    │  pool → bottleneck(512 → 1024)          → b  (B,1024, H/16, W/16) │
    ├─────────────────────────────────────────────────────────────────────┤
    │  DECODER  (Upsample → AttentionGate → Concat → ResidualConv)       │
    │  up4(b)  → att4(d4,e4) → cat → dec4 → d4 (B, 512, H/8,  W/8 )   │
    │  up3(d4) → att3(d3,e3) → cat → dec3 → d3 (B, 256, H/4,  W/4 )   │
    │  up2(d3) → att2(d2,e2) → cat → dec2 → d2 (B, 128, H/2,  W/2 )   │
    │  up1(d2) → att1(d1,e1) → cat → dec1 → d1 (B,  64, H,    W   )   │
    │  final: Conv1×1(64→1)  → main_logit                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │  DEEP SUPERVISION (training only, upsampled to H×W before loss)    │
    │  ds4: Conv1×1(512→1) on d4  →  aux logit at 1/8  scale            │
    │  ds3: Conv1×1(256→1) on d3  →  aux logit at 1/4  scale            │
    │  ds2: Conv1×1(128→1) on d2  →  aux logit at 1/2  scale            │
    └─────────────────────────────────────────────────────────────────────┘

    Forward:
        Training  → tuple (main_logit, ds4, ds3, ds2),  all (B,1,H,W)
        Eval      → main_logit only,  shape (B,1,H,W)

    Args:
        dropout_p : Dropout2d probability inside each ResidualConv block.
                    Recommended range: 0.05 – 0.15.
    """

    def __init__(self, dropout_p: float = 0.1):
        super().__init__()
        dp = dropout_p

        # ── Encoder ───────────────────────────────────────────────────
        self.enc1 = ResidualConv(3,   64,   dp)
        self.enc2 = ResidualConv(64,  128,  dp)
        self.enc3 = ResidualConv(128, 256,  dp)
        self.enc4 = ResidualConv(256, 512,  dp)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = ResidualConv(512, 1024, dp)

        # ── Decoder stage 4  (1/8 → 1/8 resolution) ──────────────────
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = ResidualConv(1024, 512, dp)   # 512 (up) + 512 (skip)

        # ── Decoder stage 3  (1/8 → 1/4 resolution) ──────────────────
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = ResidualConv(512, 256, dp)

        # ── Decoder stage 2  (1/4 → 1/2 resolution) ──────────────────
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = ResidualConv(256, 128, dp)

        # ── Decoder stage 1  (1/2 → full resolution) ─────────────────
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = ResidualConv(128, 64, dp)

        # ── Final segmentation head ───────────────────────────────────
        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # ── Deep supervision heads ────────────────────────────────────
        self.ds4 = nn.Conv2d(512, 1, kernel_size=1)
        self.ds3 = nn.Conv2d(256, 1, kernel_size=1)
        self.ds2 = nn.Conv2d(128, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming (He) normal initialisation for Conv2d.
        Constant 1/0 initialisation for BatchNorm weight/bias.
        Standard best practice for ReLU-based deep networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + attention gates
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        main_out = self.final(d1)

        if self.training:
            up = lambda t: F.interpolate(t, size=(H, W),
                                         mode='bilinear', align_corners=False)
            return main_out, up(self.ds4(d4)), up(self.ds3(d3)), up(self.ds2(d2))

        return main_out


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str,
               device:          str   = 'cpu',
               dropout_p:       float = 0.1) -> ARCUNet:
    """
    Load a trained ARCUNet from a .pt / .pth checkpoint file.

    Supports three checkpoint formats:
        • {'model_state_dict': ..., ...}   (saved by save_model / training loop)
        • {'state_dict': ...}              (legacy format)
        • raw state_dict tensor dict       (bare torch.save of model.state_dict())

    Args:
        checkpoint_path : path to checkpoint file
        device          : 'cpu' or 'cuda'
        dropout_p       : must match the value used during training

    Returns:
        ARCUNet in eval mode, moved to device.
    """
    model = ARCUNet(dropout_p=dropout_p)
    dev   = torch.device(device)

    try:
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        print("✓ Checkpoint loaded safely (weights_only=True)")
    except Exception:
        print("⚠ Retrying with weights_only=False …")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        print("✓ Checkpoint loaded (weights_only=False)")

    if isinstance(ckpt, dict):
        state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    else:
        state = ckpt

    model.load_state_dict(state)
    model.to(dev).eval()
    return model


def save_model(model:     ARCUNet,
               save_path: str,
               optimizer  = None,
               scheduler  = None,
               epoch:     int   = None,
               val_dice:  float = None,
               val_iou:   float = None) -> None:
    """
    Save an ARCUNet checkpoint.

    Always saves model weights. Optimizer, scheduler, epoch and
    validation metrics are included only when passed.

    Args:
        model      : ARCUNet instance
        save_path  : file path (.pt or .pth)
        optimizer  : torch optimizer (optional)
        scheduler  : lr scheduler (optional)
        epoch      : current epoch number (optional)
        val_dice   : validation Dice score for record-keeping (optional)
        val_iou    : validation IoU score (optional)
    """
    ckpt = {'model_state_dict': model.state_dict()}
    if optimizer is not None: ckpt['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None: ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if epoch     is not None: ckpt['epoch']    = epoch
    if val_dice  is not None: ckpt['val_dice'] = val_dice
    if val_iou   is not None: ckpt['val_iou']  = val_iou
    torch.save(ckpt, save_path)
    print(f"  ✓ Checkpoint saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict(model:        ARCUNet,
            image_tensor: torch.Tensor,
            device:       str   = 'cpu',
            threshold:    float = 0.5):
    """
    Run inference on a single preprocessed image tensor.

    The model is automatically set to eval mode (deep supervision disabled).

    Args:
        model        : trained ARCUNet
        image_tensor : shape (C,H,W) or (1,C,H,W), normalised float tensor
        device       : 'cpu' or 'cuda'
        threshold    : binarisation threshold; tune on validation set
                       (typical optimal range: 0.45 – 0.55)

    Returns:
        numpy array of shape (H, W) with values in {0.0, 1.0}
    """
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        mask = (torch.sigmoid(model(image_tensor)) > threshold).float()

    return mask.squeeze().cpu().numpy()


def predict_proba(model:        ARCUNet,
                  image_tensor: torch.Tensor,
                  device:       str = 'cpu'):
    """
    Return the raw probability map (before thresholding).

    Use this for:
        • Visualising prediction confidence
        • Threshold-tuning on a validation set
        • Soft ensemble of multiple models

    Args:
        model        : trained ARCUNet
        image_tensor : shape (C,H,W) or (1,C,H,W)
        device       : 'cpu' or 'cuda'

    Returns:
        numpy array of shape (H, W) with values in [0.0, 1.0]
    """
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(image_tensor))

    return prob.squeeze().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  QUICK SELF-TEST   (run:  python ARCUNet.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    dev    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ARCUNet(dropout_p=0.1).to(dev)
    dummy  = torch.randn(2, 3, 512, 512, device=dev)
    target = torch.randint(0, 2, (2, 1, 512, 512), device=dev).float()

    # Training mode ──────────────────────────────────────────────────────────
    model.train()
    out_train = model(dummy)
    crit  = DeepSupervisionLoss(aux_weight=0.4)
    loss  = crit(out_train, target)
    print(f"[TRAIN] output shapes : {[list(o.shape) for o in out_train]}")
    print(f"[TRAIN] DS+Combo Loss : {loss.item():.4f}")

    # Eval mode ──────────────────────────────────────────────────────────────
    model.eval()
    out_eval = model(dummy)
    print(f"\n[EVAL]  output shape  : {list(out_eval.shape)}")
    print(f"[EVAL]  Dice          : {dice_coef(out_eval, target).item():.4f}")
    print(f"[EVAL]  IoU           : {jaccard_index(out_eval, target).item():.4f}")
    print(f"[EVAL]  Accuracy      : {pixel_accuracy(out_eval, target).item():.4f}")

    # Parameter count ────────────────────────────────────────────────────────
    n = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters      : {n:,}")
