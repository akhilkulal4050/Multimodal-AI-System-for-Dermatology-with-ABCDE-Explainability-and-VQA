"""
SLRC – Skin Lesion ROI Cropping
================================
Designed as the bridge between ARCUNet (SLS stage) and the downstream
Hierarchical Classifier (HC).

ARCUNet contract (from ARCUNet_Train2.ipynb):
  • INPUT  fed to ARCUNet : RGB uint8, resized to 512×512, hair-removed,
                            then normalised with ImageNet stats and packed
                            as a float32 torch.Tensor (C,H,W).
  • OUTPUT from ARCUNet   : raw logit tensor (B,1,512,512); call sigmoid
                            then threshold (default 0.5 or best_thresh
                            found on val set) to obtain a binary mask.
                            The mask is therefore (H=512, W=512), float32,
                            values in {0.0, 1.0}.

SLRC sits AFTER ARCUNet's sigmoid+threshold step and BEFORE the HC.
It receives:
  • original_image  – the original RGB uint8 image (any resolution)
  • lesion_mask     – ARCUNet's thresholded output, either:
                        a) float32 {0,1} tensor converted to numpy, OR
                        b) uint8 {0,255} numpy array

It returns a 224×224 RGB crop ready for the HC.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def slrc_single_image(
    original_image: np.ndarray,
    lesion_mask: np.ndarray,
    output_size: tuple = (224, 224),
    padding: int = 10,
    min_area: int = 100,
    use_convex_hull: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    SLRC – Skin Lesion ROI Cropping (Single Image)

    Bridges ARCUNet's binary segmentation mask to a clean, square,
    aspect-ratio-preserving crop for the Hierarchical Classifier (HC).

    Parameters
    ----------
    original_image : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
        This is the *original* image, NOT the 512x512 ARCUNet input.
    lesion_mask : np.ndarray
        Binary mask of shape (H_m, W_m) output by ARCUNet after
        sigmoid + threshold.  Accepted dtypes:
          float32/float64 with values in [0,1]  or
          uint8 with values in {0,255} or {0,1}.
        If H_m/W_m differ from the original image size (e.g. mask is
        512x512 from ARCUNet but image is 1022x767), the mask is
        automatically upsampled with nearest-neighbour interpolation so
        bounding-box coordinates map back to original-image pixels.
    output_size : tuple[int, int]
        (width, height) of the HC input. Default (224, 224).
    padding : int
        Extra pixels added around the bounding box on all sides.
        Increased default to 10 (was 5) to capture full lesion border,
        which ARCUNet is specifically optimised to detect accurately.
    min_area : int
        Minimum pixel area (in mask space) for a connected component
        to be treated as a lesion. Default 100 – rejects tiny artefacts
        that survive morphological cleaning.
    use_convex_hull : bool
        If True, fit a convex hull over the largest lesion blob before
        computing the bounding box. This gives a tighter bbox for
        concave lesion shapes and removes internal holes in the mask.
        Default True.

    Returns
    -------
    roi_resized : np.ndarray
        uint8 RGB array of shape (output_size[1], output_size[0], 3),
        ready to be fed into the HC (after your own normalisation step).
    bbox : tuple[int, int, int, int]
        (x, y, w, h) bounding box in *original image* pixel coordinates.
    """

    # ------------------------------------------------------------------
    # Step 1: Validate inputs
    # ------------------------------------------------------------------
    if original_image.ndim != 3 or original_image.shape[2] != 3:
        raise ValueError("original_image must be an RGB array of shape (H, W, 3).")
    if lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be a 2-D array of shape (H, W).")

    img_h, img_w = original_image.shape[:2]

    # ------------------------------------------------------------------
    # Step 2: Normalise mask to uint8 binary {0, 1}
    # ARCUNet outputs logits -> sigmoid -> threshold, so mask values are
    # either float {0.0, 1.0} or already uint8 {0, 255}.
    # ------------------------------------------------------------------
    if lesion_mask.dtype in (np.float32, np.float64):
        # float path: values already in [0,1] after sigmoid+threshold
        binary_mask = (lesion_mask > 0.5).astype(np.uint8)
    else:
        # uint8 path: handle both {0,255} and {0,1}
        binary_mask = (lesion_mask > 127).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 3: Upsample mask to original image resolution if needed
    # ARCUNet always produces 512x512 output; original images vary.
    # Using INTER_NEAREST preserves hard binary edges exactly.
    # ------------------------------------------------------------------
    if binary_mask.shape != (img_h, img_w):
        binary_mask = cv2.resize(
            binary_mask, (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        )

    # ------------------------------------------------------------------
    # Step 4: Morphological cleaning
    # 5x5 kernel (larger than before) because ARCUNet border artefacts
    # can span several pixels after upsampling from 512px space.
    # OPEN removes noise speckles; CLOSE fills small holes inside lesion.
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ------------------------------------------------------------------
    # Step 5: Connected components – find all lesion candidates
    # ------------------------------------------------------------------
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    foreground_stats = stats[1:]  # skip background label 0
    valid = foreground_stats[:, cv2.CC_STAT_AREA] >= min_area

    # ------------------------------------------------------------------
    # Step 6: Fallback – centre-square crop when mask is empty
    # Matches ARCUNet's own no-lesion edge case behaviour.
    # ------------------------------------------------------------------
    if not valid.any():
        return _center_square_crop(original_image, output_size)

    # ------------------------------------------------------------------
    # Step 7: Select the largest blob
    # ------------------------------------------------------------------
    valid_indices = np.where(valid)[0]
    largest_idx   = np.argmax(foreground_stats[valid_indices, cv2.CC_STAT_AREA])
    best_stat     = foreground_stats[valid_indices[largest_idx]]
    bx = int(best_stat[cv2.CC_STAT_LEFT])
    by = int(best_stat[cv2.CC_STAT_TOP])
    bw = int(best_stat[cv2.CC_STAT_WIDTH])
    bh = int(best_stat[cv2.CC_STAT_HEIGHT])

    # ------------------------------------------------------------------
    # Step 8 (optional): Convex-hull bounding box
    # Tightens the bbox for concave lesion shapes and fills internal
    # holes – useful when ARCUNet under-segments the lesion interior,
    # which is common for darker lesion types in the ISIC datasets.
    # ------------------------------------------------------------------
    if use_convex_hull:
        blob_roi    = binary_mask[by:by + bh, bx:bx + bw]
        contours, _ = cv2.findContours(blob_roi, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_pts = np.vstack(contours)
            hull    = cv2.convexHull(all_pts)
            rx, ry, rw, rh = cv2.boundingRect(hull)
            # hull coords are relative to blob_roi; shift back to full image
            bx, by, bw, bh = bx + rx, by + ry, rw, rh

    # ------------------------------------------------------------------
    # Step 9: Padding, clamped to image bounds
    # Larger default (10 px) preserves lesion border context, which is
    # the region ARCUNet is explicitly optimised to segment accurately.
    # ------------------------------------------------------------------
    x1 = max(bx - padding, 0)
    y1 = max(by - padding, 0)
    x2 = min(bx + bw + padding, img_w)
    y2 = min(by + bh + padding, img_h)

    # ------------------------------------------------------------------
    # Step 10: Crop original-resolution colour image
    # ------------------------------------------------------------------
    roi = original_image[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Step 11: Aspect-ratio-preserving resize with letter-boxing
    # ------------------------------------------------------------------
    roi_resized = _resize_preserve_aspect(roi, output_size)

    return roi_resized, (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Convenience wrapper: accepts ARCUNet logit tensor directly
# ---------------------------------------------------------------------------

def slrc_from_logits(
    original_image: np.ndarray,
    logit_tensor,
    threshold: float = 0.5,
    output_size: tuple = (224, 224),
    padding: int = 10,
    min_area: int = 100,
    use_convex_hull: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Convenience wrapper that accepts ARCUNet's raw logit tensor and
    handles sigmoid + threshold + numpy conversion internally.

    Parameters
    ----------
    logit_tensor : torch.Tensor
        Raw output from ARCUNet forward pass.
        Accepted shapes: (1,1,H,W), (1,H,W), or (H,W).
    threshold : float
        Sigmoid threshold. Pass the best_thresh found by ARCUNet_Train2's
        val-set threshold-search cell for best results. Default 0.5.

    All other parameters are forwarded to slrc_single_image.

    Example
    -------
    >>> with torch.no_grad():
    ...     logits = model(img_tensor)          # (1, 1, 512, 512)
    >>> roi, bbox = slrc_from_logits(
    ...     original_rgb_image, logits,
    ...     threshold=best_thresh               # from ARCUNet_Train2 Cell 9
    ... )
    """
    import torch

    with torch.no_grad():
        prob = torch.sigmoid(logit_tensor.float())

    mask_np = prob.squeeze().cpu().numpy()            # (H, W), float32 in [0,1]
    binary  = (mask_np > threshold).astype(np.uint8)  # {0, 1}

    return slrc_single_image(
        original_image,
        binary,
        output_size=output_size,
        padding=padding,
        min_area=min_area,
        use_convex_hull=use_convex_hull,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resize_preserve_aspect(
    image: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize image to output_size preserving aspect ratio.
    Shorter axis is letter-boxed with black (zero) padding.
    Uses INTER_AREA for downscaling (sharper) and INTER_LINEAR for upscaling.
    """
    target_w, target_h = output_size
    h, w = image.shape[:2]

    scale  = min(target_w / w, target_h / h)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    new_w  = int(round(w * scale))
    new_h  = int(round(h * scale))

    resized  = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas   = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    pad_top  = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas


def _center_square_crop(
    image: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Fallback: centre-square crop when ARCUNet finds no valid lesion region.
    Returns (roi_resized, bbox) matching the slrc_single_image signature.
    """
    H, W  = image.shape[:2]
    dim   = min(H, W)
    x1_fb = (W - dim) // 2
    y1_fb = (H - dim) // 2
    roi   = image[y1_fb:y1_fb + dim, x1_fb:x1_fb + dim]
    return _resize_preserve_aspect(roi, output_size), (x1_fb, y1_fb, dim, dim)