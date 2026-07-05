"""
Stage 7 - Stage6QADataset PyTorch Dataset.

Each sample returns:
    image_tensor   : (3, 336, 336) full image with alpha-blended bbox overlay
    roi_tensor     : (3, 336, 336) CLIP-style ROI crop from bbox
    bbox           : FloatTensor [x1, y1, x2, y2]
    question       : str
    answer         : str
    question_type  : str  (open / close / multichoice / region)
    clinical_summary: str
    image_path     : str
    disease_label  : str

This class is used by Component 2 (R-LLaVA training) and Component 5 (testing).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Stage6QADataset(Dataset):
    """
    PyTorch Dataset for Stage6QA parquets.
    Returns both the alpha-blended full image (for R-LLaVA) and the
    CLIP-style ROI crop (for spatial grounding verification).
    """

    IMG_SIZE = 336    # CLIP-ViT-L/14@336px, matches LLaVA-v1.6

    def __init__(self, parquet_path, img_root=None, question_type=None,
                 split=None, alpha_mode='dynamic'):
        import pandas as pd
        self.df = pd.read_parquet(parquet_path)

        if question_type:
            self.df = self.df[self.df['question_type'] == question_type].reset_index(drop=True)
        if split:
            if 'split' in self.df.columns:
                self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.img_root   = Path(img_root) if img_root else None
        self.alpha_mode = alpha_mode

        self.transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f'Stage6QADataset: {len(self.df):,} samples', end='')
        if question_type: print(f' (type={question_type})', end='')
        if split:         print(f' (split={split})', end='')
        print()

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_path):
        try:
            path = (self.img_root / Path(image_path).name) if self.img_root else Path(image_path)
            return Image.open(path).convert('RGB')
        except Exception:
            return Image.new('RGB', (self.IMG_SIZE, self.IMG_SIZE), (128, 128, 128))

    def _parse_bbox(self, raw):
        if isinstance(raw, (list, np.ndarray)):
            return [int(v) for v in raw]
        s = str(raw).strip().replace('(','').replace(')','').replace('[','').replace(']','')
        parts = [p.strip() for p in s.split(',') if p.strip()]
        if len(parts) != 4:
            parts = [p.strip() for p in s.split() if p.strip()]
        return [int(float(p)) for p in parts]

    def _alpha_blend(self, img, bbox):
        """
        R-LLaVA merged image: draw bbox rectangle on full image.
        alpha controls overlay strength (dynamic jitter during training).
        """
        arr = np.array(img)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        overlay = arr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
        import random
        alpha = random.uniform(96/255, 1.0) if self.alpha_mode == 'dynamic' else 200/255
        blended = cv2.addWeighted(overlay, alpha, arr, 1 - alpha, 0)
        return Image.fromarray(blended.astype('uint8'))

    def _crop_roi(self, img, bbox):
        """CLIP-style ROI crop: crop to bbox then resize."""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            roi = img.crop((x1, y1, x2, y2))
            return roi if min(roi.size) > 0 else img
        except Exception:
            return img

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        bbox = self._parse_bbox(row['bbox'])
        img  = self._load_image(row['image_path'])

        merged_img = self._alpha_blend(img, bbox)
        image_tensor = self.transform(merged_img)

        roi_img    = self._crop_roi(img, bbox)
        roi_tensor = self.transform(roi_img)

        return {
            'image_tensor'  : image_tensor,       # (3, 336, 336) blended
            'roi_tensor'    : roi_tensor,          # (3, 336, 336) ROI crop
            'bbox'          : torch.tensor(bbox, dtype=torch.float32),
            'bbox_str'      : row.get('bbox_str', str(bbox)),
            'question'      : str(row['question']),
            'answer'        : str(row['answer']),
            'question_type' : str(row['question_type']),
            'clinical_summary': str(row.get('clinical_summary', '')),
            'image_path'    : str(row['image_path']),
            'disease_label' : str(row.get('disease_label', 'unknown')),
        }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <path_to_stage6qa_dataset.parquet>')
        sys.exit(1)
    ds = Stage6QADataset(sys.argv[1])
    sample = ds[0]
    print('Keys          :', list(sample.keys()))
    print('image_tensor  :', sample['image_tensor'].shape)
    print('roi_tensor    :', sample['roi_tensor'].shape)
    print('bbox          :', sample['bbox'])
    print('question      :', sample['question'])
    print('answer        :', sample['answer'][:80])
