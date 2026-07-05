
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add MedProc module to path
MEDPROC_DIR = Path('/home/vjti-comp/Desktop/Final Project Code/MedProc')
sys.path.insert(0, str(MEDPROC_DIR))

class TextEncoder(nn.Module):
    """
    Text Encoder using trained MedProc model.
    Extracts features from medical text.
    """
    def __init__(self, checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # Load MedProc model from checkpoint
        if checkpoint_path is None:
            checkpoint_path = MEDPROC_DIR / 'checkpoints' / 'medproc_checkpoint.pt'
        
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            print(f"Loading MedProc checkpoint from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=device)
            
            # Extract metadata
            self.label_map = ckpt.get('label_map', {})
            self.symptom_keywords = ckpt.get('symptom_keywords', [])
            self.icd_threshold = ckpt.get('icd_threshold', 0.35)
            self.sym_threshold = ckpt.get('sym_threshold', 0.3)
            
            # Initialize MedProc model
            self._init_medproc_model(ckpt)
            
            self.medproc_model.eval()
            print(f"✓ MedProc model loaded successfully")
            print(f"  - ICD classes: {len(self.label_map)}")
        else:
            print(f"⚠ MedProc checkpoint not found at {checkpoint_path}")
            print(f"  Using placeholder encoder. Train MedProc model first.")
            self.medproc_model = None
            self.tokenizer = None
            
        # Feature dim: 768-dim raw [CLS] pooler output
        # No projection — FusionClassifier trained on 768-dim (TEXT_DIM=768 in train_it_fusion.ipynb)
        self.feature_dim = 768

    def _init_medproc_model(self, ckpt):
        """Initialize MedProc model from checkpoint."""
        try:
            from medproc_model import MedProcMultiTaskModel, MODEL_NAME
            from transformers import AutoTokenizer
            
            # Get tokenizer
            self.tokenizer = ckpt.get('tokenizer')
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Initialize model
            num_icd_labels = len(ckpt.get('label_map', {}))
            self.medproc_model = MedProcMultiTaskModel(
                model_name=MODEL_NAME,
                num_icd_labels=num_icd_labels
            ).to(self.device)
            
            # Load weights
            if 'model_state' in ckpt:
                self.medproc_model.load_state_dict(ckpt['model_state'])
            elif 'state_dict' in ckpt:
                self.medproc_model.load_state_dict(ckpt['state_dict'])
            
        except Exception as e:
            print(f"⚠ Could not initialize MedProc model: {e}")
            self.medproc_model = None
            self.tokenizer = None

    def forward(self, text):
        """
        Extract features from medical text using MedProc.
        
        Args:
            text: str or list of strings
            
        Returns:
            features: [B, 768] encoded features (raw Bio_ClinicalBERT [CLS] pooler output)
            predictions: dict with icd, symptom predictions
        """
        if self.medproc_model is None or self.tokenizer is None:
            # Placeholder: return zeros
            if isinstance(text, str):
                batch_size = 1
            else:
                batch_size = len(text)
            return torch.zeros(batch_size, 768, device=self.device), {}
        
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            # Get BERT embeddings (from pooler output)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            # Get model predictions
            icd_logits, symptom_logits = self.medproc_model(
                input_ids, attention_mask
            )
            
            # Get BERT hidden state for features
            bert_output = self.medproc_model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_features = bert_output.pooler_output  # [B, 768]
            
            # Apply thresholds for predictions
            icd_preds = torch.sigmoid(icd_logits) > self.icd_threshold
            symptom_preds = torch.sigmoid(symptom_logits) > self.sym_threshold
            
            predictions = {
                'icd_logits': icd_logits.cpu().numpy(),
                'icd_preds': icd_preds.cpu().numpy(),
                'symptom_logits': symptom_logits.cpu().numpy(),
                'symptom_detected': symptom_preds.cpu().numpy(),
                'icd_classes': self.label_map,
                'symptom_keywords': self.symptom_keywords,
            }
        
        return pooled_features, predictions  # [B, 768] — matches TEXT_DIM=768 in FusionClassifier
