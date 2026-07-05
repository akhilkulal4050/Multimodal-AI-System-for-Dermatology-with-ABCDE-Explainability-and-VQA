"""
MedProc Model Module
Defines the multi-task model architecture and training utilities.

Tasks:
  1. ICD-9 multi-label classification (sigmoid)
  2. Symptom / Evolution relevance detection (binary)

Note: Drug recommendation is handled downstream by drug_rules.py using HC
output (lesion class + ABCDE risk). MedProc is NOT responsible for drug
detection — only for supplying the Evolution (E) score and symptom keywords
to IT-Fusion (Stage 6).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
CHECKPOINT_DIR = Path('/data/Stagewise Dataset/MedProc/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 512   # Bio_ClinicalBERT maximum
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 2e-5

# Task loss weights
WEIGHT_ICD     = 1.0   # Primary task — ICD-9 multi-label classification
WEIGHT_SYMPTOM = 0.5   # Secondary task — symptom/evolution relevance


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class MedProcMultiTaskModel(nn.Module):
    """
    Two-task learning model with Bio_ClinicalBERT backbone.

    Tasks
    -----
    1. ICD-9 multi-label classification (sigmoid + BCEWithLogitsLoss)
       → identifies relevant disease categories from the clinical note
    2. Symptom / Evolution relevance (binary sigmoid)
       → signals whether the note contains clinically relevant symptom
         descriptions that should contribute to the ABCDE Evolution (E) score

    Architecture
    ------------
    Bio_ClinicalBERT (last 3 encoder layers + pooler trainable)
        └─ [CLS] pooler_output (768-dim)
               ├─ icd_head     → [B, num_icd_labels]
               └─ symptom_head → [B, 1]

    Only the last 3 encoder layers and the pooler are fine-tuned (~15% of
    total parameters). All earlier BERT weights remain frozen.
    """

    def __init__(
        self,
        model_name:     str = MODEL_NAME,
        num_icd_labels: int = 200,
        dropout:        float = 0.3,
    ):
        super().__init__()

        self.bert   = AutoModel.from_pretrained(model_name)
        hidden      = self.bert.config.hidden_size          # 768
        n_layers    = self.bert.config.num_hidden_layers    # 12

        # Freeze everything first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze last 3 encoder transformer blocks
        for layer in self.bert.encoder.layer[n_layers - 3:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze pooler
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        self.drop = nn.Dropout(dropout)

        # ── Head 1: ICD-9 multi-label classification
        self.icd_head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_icd_labels),
        )

        # ── Head 2: Symptom / Evolution relevance (binary)
        self.symptom_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        icd_logits     : [B, num_icd_labels]  — raw (pre-sigmoid) ICD scores
        symptom_logits : [B, 1]               — raw symptom/evolution relevance
        """
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.drop(out.pooler_output)       # [B, 768]

        icd_logits     = self.icd_head(pooled)      # [B, num_icd_labels]
        symptom_logits = self.symptom_head(pooled)  # [B, 1]

        return icd_logits, symptom_logits

    def get_trainable_params_count(self) -> Tuple[int, int]:
        """Return (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MedProcDataset(Dataset):
    """
    PyTorch Dataset for medical text.

    Parameters
    ----------
    texts           : list of raw clinical note strings
    icd_labels      : np.ndarray [N, num_icd_labels]  — multi-hot ICD labels
    symptom_labels  : list[int]  — 1 if note has relevant symptoms, else 0
    indices         : list[int]  — which rows of texts/labels to use
    tokenizer       : HuggingFace tokenizer
    max_seq_len     : int        — truncation length (default MAX_SEQ_LEN)

    Returns (per item)
    ------------------
    input_ids      : [max_seq_len]
    attention_mask : [max_seq_len]
    icd_labels     : [num_icd_labels]   FloatTensor
    symptom_label  : [1]                FloatTensor
    """

    def __init__(
        self,
        texts:          list,
        icd_labels:     np.ndarray,
        symptom_labels: list,
        indices:        list,
        tokenizer,
        max_seq_len:    int = MAX_SEQ_LEN,
    ):
        self.texts          = [texts[i] for i in indices]
        self.icd_labels     = icd_labels[list(indices)]
        self.symptom_labels = [symptom_labels[i] for i in indices]
        self.tokenizer      = tokenizer
        self.max_seq_len    = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids':     enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'icd_labels':    torch.FloatTensor(self.icd_labels[idx]),
            'symptom_label': torch.FloatTensor([self.symptom_labels[idx]]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    texts:          list,
    icd_labels:     np.ndarray,
    symptom_labels: list,
    train_idx:      list,
    val_idx:        list,
    tokenizer,
    batch_size:     int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Parameters
    ----------
    texts          : list of clinical note strings
    icd_labels     : np.ndarray [N, num_icd_labels]
    symptom_labels : list[int]  — 1 if note has relevant symptoms, else 0
                     (derived from has_symptom or evolution keyword presence)
    train_idx      : list[int]  — training row indices
    val_idx        : list[int]  — validation row indices
    tokenizer      : HuggingFace tokenizer
    batch_size     : int

    Returns
    -------
    (train_dataloader, val_dataloader)
    """
    train_ds = MedProcDataset(texts, icd_labels, symptom_labels, train_idx, tokenizer)
    val_ds   = MedProcDataset(texts, icd_labels, symptom_labels, val_idx,   tokenizer)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dl, val_dl


# ─────────────────────────────────────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:          nn.Module,
    loader:         DataLoader,
    optimizer,
    scheduler,
    icd_loss_fn,
    sym_loss_fn,
    weight_icd:     float = WEIGHT_ICD,
    weight_symptom: float = WEIGHT_SYMPTOM,
    device                = DEVICE,
) -> float:
    """
    Train for one epoch.

    Returns
    -------
    Average combined loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        icd_labels_b   = batch['icd_labels'].to(device)
        sym_labels_b   = batch['symptom_label'].to(device)

        optimizer.zero_grad()
        icd_logits, sym_logits = model(input_ids, attention_mask)

        loss = (
            weight_icd     * icd_loss_fn(icd_logits, icd_labels_b) +
            weight_symptom * sym_loss_fn(sym_logits, sym_labels_b)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model:          nn.Module,
    loader:         DataLoader,
    icd_loss_fn,
    sym_loss_fn,
    weight_icd:     float = WEIGHT_ICD,
    weight_symptom: float = WEIGHT_SYMPTOM,
    device                = DEVICE,
) -> Tuple[float, float]:
    """
    Evaluate for one epoch.

    Returns
    -------
    (avg_loss, symptom_accuracy)
      avg_loss         — weighted combined loss
      symptom_accuracy — fraction of symptom/evolution labels predicted correctly
    """
    model.eval()
    total_loss    = 0.0
    correct_sym   = 0
    total         = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        icd_labels_b   = batch['icd_labels'].to(device)
        sym_labels_b   = batch['symptom_label'].to(device)

        icd_logits, sym_logits = model(input_ids, attention_mask)

        loss = (
            weight_icd     * icd_loss_fn(icd_logits, icd_labels_b) +
            weight_symptom * sym_loss_fn(sym_logits, sym_labels_b)
        )
        total_loss += loss.item()

        preds        = (torch.sigmoid(sym_logits) > 0.5).float()
        correct_sym += (preds == sym_labels_b).sum().item()
        total       += len(sym_labels_b)

    return total_loss / len(loader), correct_sym / total


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:            nn.Module,
    tokenizer,
    label_map:        Dict,
    symptom_keywords: list,
    thresholds:       Dict,
    output_path:      str = str(CHECKPOINT_DIR / 'medproc_checkpoint.pt'),
):
    """
    Save model weights, tokenizer, and metadata to a single checkpoint.

    Saved keys
    ----------
    model_state       : model.state_dict()
    tokenizer         : HuggingFace tokenizer object
    label_map         : {int_index -> icd_prefix_string}
    symptom_keywords  : list[str] from medproc_dataset.SYMPTOM_KEYWORDS
    icd_threshold     : float  — min confidence to report an ICD prediction
    sym_threshold     : float  — min confidence to flag symptom relevance
    """
    checkpoint = {
        'model_state':      model.state_dict(),
        'tokenizer':        tokenizer,
        'label_map':        label_map,
        'symptom_keywords': symptom_keywords,
        'icd_threshold':    thresholds.get('icd',     0.35),
        'sym_threshold':    thresholds.get('symptom', 0.30),
    }
    torch.save(checkpoint, output_path)
    print(f'✓ Checkpoint saved to {output_path}')


def load_checkpoint(
    checkpoint_path: str = str(CHECKPOINT_DIR / 'medproc_checkpoint.pt'),
    device               = DEVICE,
) -> Dict:
    """Load and return checkpoint dict."""
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'MedProc Model Module')
    print(f'  Device     : {DEVICE}')
    print(f'  Backbone   : {MODEL_NAME}')
    print(f'  Max seq len: {MAX_SEQ_LEN}')
    print(f'  Tasks      : ICD classification + Symptom/Evolution relevance')
    print(f'  Note       : Drug recommendation is handled by drug_rules.py '
          f'(uses HC output, not MedProc)')