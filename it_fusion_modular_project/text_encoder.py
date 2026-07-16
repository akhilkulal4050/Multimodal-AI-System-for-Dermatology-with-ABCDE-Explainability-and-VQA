
import torch
import torch.nn as nn
from pathlib import Path
import sys

MEDPROC_DIR = Path('/home/vjti-comp/Desktop/Final Project Code/MedProc')
sys.path.insert(0, str(MEDPROC_DIR))

class TextEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        if checkpoint_path is None:
            checkpoint_path = MEDPROC_DIR / 'checkpoints' / 'medproc_checkpoint.pt'
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            print(f"Loading MedProc checkpoint from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.label_map        = ckpt.get('label_map', {})
            self.symptom_keywords = ckpt.get('symptom_keywords', [])
            self.icd_threshold    = ckpt.get('icd_threshold', 0.35)
            self.sym_threshold    = ckpt.get('sym_threshold', 0.30)
            self._init_medproc_model(ckpt)
            self.medproc_model.eval()
            print(f"✓ MedProc model loaded successfully")
            print(f"  - ICD classes: {len(self.label_map)}")
        else:
            print(f"⚠ MedProc checkpoint not found at {checkpoint_path}")
            self.medproc_model    = None
            self.tokenizer        = None
            self.label_map        = {}
            self.symptom_keywords = []
            self.icd_threshold    = 0.35
            self.sym_threshold    = 0.30
        self.feature_dim = 768

    def _init_medproc_model(self, ckpt):
        try:
            from medproc_model import MedProcMultiTaskModel, MODEL_NAME
            from transformers import AutoTokenizer
            self.tokenizer = ckpt.get('tokenizer')
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            num_icd_labels = len(ckpt.get('label_map', {}))
            self.medproc_model = MedProcMultiTaskModel(
                model_name=MODEL_NAME, num_icd_labels=num_icd_labels
            ).to(self.device)
            if 'model_state' in ckpt:
                self.medproc_model.load_state_dict(ckpt['model_state'])
            elif 'state_dict' in ckpt:
                self.medproc_model.load_state_dict(ckpt['state_dict'])
        except Exception as e:
            print(f"⚠ Could not initialize MedProc model: {e}")
            self.medproc_model = None
            self.tokenizer     = None

    def forward(self, text):
        if self.medproc_model is None or self.tokenizer is None:
            batch_size = 1 if isinstance(text, str) else len(text)
            return torch.zeros(batch_size, 768, device=self.device), {
                'evolution_score': 0.0, 'symptom_keywords': [],
                'symptom_detected': False, 'top_icd_prefix': 'unknown'
            }
        if isinstance(text, str):
            text = [text]
        encoded = self.tokenizer(
            text, max_length=512, truncation=True,
            padding='max_length', return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            input_ids      = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            icd_logits, symptom_logits = self.medproc_model(input_ids, attention_mask)
            bert_out        = self.medproc_model.bert(
                input_ids=input_ids, attention_mask=attention_mask)
            pooled_features = bert_out.pooler_output
            icd_probs       = torch.sigmoid(icd_logits)
            symptom_score   = torch.sigmoid(symptom_logits).squeeze().item()
            top_idx         = int(icd_probs.argmax(dim=-1).item())
            top_icd         = self.label_map.get(str(top_idx),
                              self.label_map.get(top_idx, 'unknown'))
            predictions = {
                'icd_logits':       icd_logits.cpu().numpy(),
                'icd_preds':        (icd_probs > self.icd_threshold).cpu().numpy(),
                'symptom_logits':   symptom_logits.cpu().numpy(),
                'symptom_detected': symptom_score >= self.sym_threshold,
                'evolution_score':  round(symptom_score, 4),
                'symptom_keywords': self.symptom_keywords,
                'top_icd_prefix':   top_icd,
                'icd_classes':      self.label_map,
            }
        return pooled_features, predictions
