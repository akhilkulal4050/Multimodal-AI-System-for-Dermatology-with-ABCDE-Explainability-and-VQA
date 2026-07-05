#!/usr/bin/env python3
"""
NutriDermAI — R-LLaVA Stage I + Stage II Training
===================================================
Self-contained script: sets up environment, then runs Stage I + Stage II.

Usage:
    python nutriderm_rllava_train.py

Or background (survives SSH disconnect):
    nohup python -u nutriderm_rllava_train.py > /home/user_name/training.log 2>&1 &
    tail -f /home/user_name/training.log

Environment is set up automatically on first run. Packages are installed
directly into the running Python — no venv needed when running inside the
nvcr.io/nvidia/pytorch:23.10-py3 container.

Checkpoints are saved to /data/Stagewise Dataset/Stage7/rllava_checkpoints/
Training log + curves saved to /data/Stagewise Dataset/Stage7/
"""

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — ENVIRONMENT SETUP
# Runs pip installs inline. Safe to re-run — pip skips already-installed.
# ═══════════════════════════════════════════════════════════════════════════

import subprocess, sys, os
from pathlib import Path

def _pip(*packages, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"] + list(packages)
    if extra_args:
        cmd += extra_args
    r = subprocess.run(cmd, capture_output=True, text=True)
    status = "OK" if r.returncode == 0 else "FAIL"
    label  = packages[0][:45] if packages else ""
    print(f"  [{status}] {label}")
    if r.returncode != 0:
        print(f"         {r.stderr[-300:]}")
    return r.returncode == 0

print("=" * 60)
print(" Step 0: Environment Setup")
print("=" * 60)

# ── 0a: Fix torch if it's the wrong CUDA build ─────────────────────────────
_torch_check = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
    capture_output=True, text=True
)
if _torch_check.stdout.strip() != "True":
    print("  torch CUDA not working — reinstalling compatible build...")
    _pip("torch==2.3.0", "torchvision==0.18.0",
         extra_args=["--index-url", "https://download.pytorch.org/whl/cu121"])
    print("  torch reinstalled for CUDA 12.1 (matches driver 535/12.2)")
else:
    print("  [OK] torch CUDA already working")

# ── 0b: Fix transformer_engine (pre-installed, conflicts with peft) ─────────
_te = subprocess.run(
    [sys.executable, "-c", "import transformer_engine"],
    capture_output=True, text=True
)
if _te.returncode == 0:
    print("  Removing transformer_engine (conflicts with peft/transformers)...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "transformer_engine", "transformer-engine"],
                   capture_output=True)
    print("  [OK] transformer_engine removed")

# ── 0c: Pin numpy/pandas (base image ships 1.22.2 — too old for transformers) ──
_np_ver = subprocess.run(
    [sys.executable, "-c", "import numpy; print(numpy.__version__)"],
    capture_output=True, text=True
).stdout.strip()
if _np_ver < "1.26":
    print(f"  numpy {_np_ver} too old — upgrading...")
    _pip("numpy==1.26.4", "pandas==2.2.3", "pyarrow>=12.0.0")
else:
    print(f"  [OK] numpy {_np_ver}")

# ── 0d: Fix opencv ──────────────────────────────────────────────────────────
_cv2_ok = subprocess.run(
    [sys.executable, "-c", "import cv2; cv2.cvtColor(__import__('numpy').zeros((2,2,3),dtype='uint8'),cv2.COLOR_RGB2BGR)"],
    capture_output=True, text=True
)
if _cv2_ok.returncode != 0:
    print("  Fixing opencv...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "opencv-python-headless", "opencv-python"],
                   capture_output=True)
    import shutil, site
    for sp in site.getsitepackages():
        cv2_path = Path(sp) / "cv2"
        if cv2_path.exists():
            shutil.rmtree(cv2_path, ignore_errors=True)
    _pip("opencv-python-headless==4.8.1.78")
else:
    print("  [OK] opencv")

# ── 0e: Core ML packages ────────────────────────────────────────────────────
print("  Installing ML packages...")
_pip("transformers==4.44.0")
_pip("tokenizers==0.19.1")
_pip("peft==0.12.0")
_pip("accelerate==0.33.0")
_pip("bitsandbytes>=0.43.0")
_pip("huggingface_hub")
_pip("kagglehub")
_pip("pillow", "tqdm", "matplotlib", "scikit-learn")
_pip("nltk")
_pip("rouge-score==0.1.2")
_pip("absl-py")

# ── 0f: Final verification ──────────────────────────────────────────────────
print("\n  Final check:")
checks = [
    ("torch+CUDA",          "import torch; assert torch.cuda.is_available()"),
    ("numpy 1.26",          "import numpy as np; assert np.__version__ >= '1.26'"),
    ("transformers 4.44",   "import transformers; assert transformers.__version__ == '4.44.0'"),
    ("peft",                "from peft import LoraConfig, get_peft_model"),
    ("bitsandbytes GPU",    "import bitsandbytes as bnb, torch; bnb.nn.Linear4bit(4,4).to('cuda')"),
    ("LlavaNext",           "from transformers import LlavaNextForConditionalGeneration"),
    ("opencv",              "import cv2; import numpy as np; cv2.cvtColor(np.zeros((2,2,3),dtype='uint8'),cv2.COLOR_RGB2BGR)"),
]
all_ok = True
for name, code in checks:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    ok = r.returncode == 0
    print(f"    {'[OK]' if ok else '[FAIL]'} {name}")
    if not ok:
        print(f"           {r.stderr[-200:]}")
        all_ok = False

if not all_ok:
    print("\nSome checks failed. Fix issues above before continuing.")
    sys.exit(1)

print("\n  Environment OK — starting training\n")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS + CONFIG
# ═══════════════════════════════════════════════════════════════════════════

import os, json, gc, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
assert torch.cuda.is_available(), "No GPU detected"

print("=" * 60)
print(" Step 1: Config + Dataset Download")
print("=" * 60)

n_gpus = torch.cuda.device_count()
print(f"GPUs: {n_gpus}")
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    vram = torch.cuda.get_device_properties(i).total_memory / 1e9
    sm   = torch.cuda.get_device_capability(i)
    print(f"  GPU {i}: {name}  {vram:.1f} GB  SM {sm[0]}.{sm[1]}")

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR       = Path('/home/prasannam24-26/rllava')
CHECKPOINT_DIR = WORK_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── HuggingFace token ────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# HF_TOKEN = "hf_xxxxxxxxxxxx"  # uncomment and paste if needed

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("HuggingFace: logged in")
else:
    print("HuggingFace: no token set (needed only for first download)")

# ── Dataset download ─────────────────────────────────────────────────────────
print("\nDownloading dataset via kagglehub...")
import kagglehub
dataset_path = Path(kagglehub.dataset_download("akhilkulal/nutriderm-stage7-vqa"))
print(f"Dataset path: {dataset_path}")

TRAIN_PARQUET = dataset_path / 'stage7_vqa_train.parquet'
VAL_PARQUET   = dataset_path / 'stage7_vqa_val.parquet'
if not TRAIN_PARQUET.exists():
    for f in dataset_path.rglob('stage7_vqa_train.parquet'): TRAIN_PARQUET = f; break
if not VAL_PARQUET.exists():
    for f in dataset_path.rglob('stage7_vqa_val.parquet'):   VAL_PARQUET   = f; break
assert TRAIN_PARQUET.exists(), f"train parquet not found under {dataset_path}"
assert VAL_PARQUET.exists(),   f"val parquet not found under {dataset_path}"
print(f"Train: {TRAIN_PARQUET}\nVal  : {VAL_PARQUET}")

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE       = 336
GRID_PINPOINTS = [[336,672],[672,336],[672,672],[1008,336],[336,1008]]
ALPHA_MODE     = 'dynamic'
RANDOM_SEED    = 42

# Stage I
S1_SUBSAMPLE   = None   # None = all unique images
S1_LR          = 1e-3
S1_GRAD_ACCUM  = 4
S1_EPOCHS      = 3
SKIP_STAGE1    = False  # set True to skip if projector already trained

# Stage II
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.0
MAX_TRAIN_SAMPLES   = None
MAX_VAL_SAMPLES     = None
STAGE2_LR           = 2e-5
STAGE2_BATCH        = 4
STAGE2_GRAD_ACCUM   = 4
STAGE2_EPOCHS       = 7
STAGE2_WARMUP       = 0.03
STAGE2_MAX_NEW_TOK  = 64
EARLY_STOP_PATIENCE = 5

# ── Seeds ────────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# ── Load dataframes ──────────────────────────────────────────────────────────
train_df = pd.read_parquet(TRAIN_PARQUET)
val_df   = pd.read_parquet(VAL_PARQUET)
if MAX_TRAIN_SAMPLES and len(train_df) > MAX_TRAIN_SAMPLES:
    train_df = train_df.sample(MAX_TRAIN_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
if MAX_VAL_SAMPLES and len(val_df) > MAX_VAL_SAMPLES:
    val_df = val_df.sample(MAX_VAL_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"\nTrain: {len(train_df):,}  Val: {len(val_df):,}")
if 'question_type' in train_df.columns:
    print(f"Q-types: {dict(train_df['question_type'].value_counts())}")

# ── Resume detection ─────────────────────────────────────────────────────────
RESUME_FROM = None
resume_p = CHECKPOINT_DIR / 'rllava_stage2_best'
if resume_p.exists():
    RESUME_FROM = str(resume_p)
    print(f"Resume checkpoint found: {RESUME_FROM}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 2: Load Model (4-bit NF4)")
print("=" * 60)

gc.collect()
for i in range(torch.cuda.device_count()):
    torch.cuda.empty_cache()

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

# ── Model path detection ─────────────────────────────────────────────────────
CANDIDATE_PATHS = [
    Path('/home/prasannam24-26/rllava/llava-v1.6-mistral-7b-hf'),  # confirmed location
    Path('/home/prasannam24-26/rllava') / 'llava-v1.6-mistral-7b-hf',
    Path.home() / 'rllava' / 'llava-v1.6-mistral-7b-hf',
    Path.home() / 'llava-v1.6-mistral-7b-hf',
    Path('/data/models/llava-v1.6-mistral-7b-hf'),
    Path('llava-v1.6-mistral-7b-hf'),                  # relative cwd fallback
]
HF_MODEL_ID      = 'llava-hf/llava-v1.6-mistral-7b-hf'
LOCAL_MODEL_PATH = None

for p in CANDIDATE_PATHS:
    if p.exists() and any(p.glob('*.safetensors')):
        LOCAL_MODEL_PATH = p
        break

if LOCAL_MODEL_PATH is not None:
    BASE_MODEL_ID = str(LOCAL_MODEL_PATH)
    n_shards = len(list(LOCAL_MODEL_PATH.glob('*.safetensors')))
    print(f"Using LOCAL model: {LOCAL_MODEL_PATH}  ({n_shards} shards)")
    required = ['config.json', 'tokenizer_config.json', 'preprocessor_config.json']
    missing  = [f for f in required if not (LOCAL_MODEL_PATH / f).exists()]
    if missing:
        print(f"  WARNING: missing files: {missing}")
else:
    BASE_MODEL_ID = HF_MODEL_ID
    print(f"Local model not found — downloading from HuggingFace (~14 GB)")
    HF_CACHE = Path('/data/hf_cache')
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(HF_CACHE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {BASE_MODEL_ID} ...")
try:
    model = LlavaNextForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config,
        device_map="auto", low_cpu_mem_usage=True)
except OSError as e:
    print(f"Network error: {e}\nRetrying offline...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config,
        device_map="auto", low_cpu_mem_usage=True, local_files_only=True)

model.config.use_cache = False
model.enable_input_require_grads()

processor = LlavaNextProcessor.from_pretrained(BASE_MODEL_ID)
text_tok   = processor.tokenizer
if text_tok.pad_token is None:
    text_tok.pad_token    = text_tok.eos_token
    text_tok.pad_token_id = text_tok.eos_token_id

for i in range(torch.cuda.device_count()):
    used = torch.cuda.memory_allocated(i) / 1e9
    free = (torch.cuda.get_device_properties(i).total_memory
            - torch.cuda.memory_allocated(i)) / 1e9
    print(f"  GPU {i}: {used:.2f} GB used | {free:.2f} GB free")
print("Model loaded.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — IMAGE + TOKEN UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

from transformers.models.llava_next.image_processing_llava_next import select_best_resolution

IMAGE_TOKEN_ID = 32000
_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3,1,1)
_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)

def encode_images(images):
    def _tile(img, h, w):
        a = np.array(img.resize((w,h)), dtype=np.float32).transpose(2,0,1) / 255.
        return (torch.from_numpy(a) - _MEAN) / _STD
    all_tiles, sizes = [], []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.new('RGB', (IMG_SIZE,IMG_SIZE), (128,128,128))
        img = img.convert('RGB')
        bh, bw = select_best_resolution((img.height,img.width), GRID_PINPOINTS)
        tiles  = []; resized = img.resize((bw,bh))
        for r in range(0, bh, IMG_SIZE):
            for c in range(0, bw, IMG_SIZE):
                tiles.append(_tile(resized.crop((c,r,c+IMG_SIZE,r+IMG_SIZE)), IMG_SIZE, IMG_SIZE))
        tiles.append(_tile(img, IMG_SIZE, IMG_SIZE))
        all_tiles.append(tiles); sizes.append([bh, bw])
    max_t = max(len(t) for t in all_tiles)
    pv    = torch.zeros(len(images), max_t, 3, IMG_SIZE, IMG_SIZE)
    for i, tiles in enumerate(all_tiles):
        for j, t in enumerate(tiles): pv[i,j] = t
    return pv, torch.tensor(sizes, dtype=torch.long)

try:
    vis_cfg = getattr(model.config, 'vision_config', None)
    ps      = getattr(vis_cfg, 'patch_size', 14) if vis_cfg else 14
    isz     = getattr(vis_cfg, 'image_size', 336) if vis_cfg else 336
    TOKENS_PER_PATCH = (isz // ps) ** 2
    if TOKENS_PER_PATCH == 576: TOKENS_PER_PATCH = 584
except Exception:
    TOKENS_PER_PATCH = 584
print(f"TOKENS_PER_PATCH = {TOKENS_PER_PATCH}")

try:
    from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches as _r
    def _n_patches(sz, gp, ps):
        s = sz[0] if (isinstance(sz,list) and isinstance(sz[0],list)) else sz
        return _r(s, gp, ps)
except ImportError:
    def _n_patches(sz, gp, ps):
        s = sz[0] if (isinstance(sz,list) and isinstance(sz[0],list)) else sz
        return (s[0]//ps) * (s[1]//ps) + 1

def build_input_ids(text, image_sizes_list, max_len=2048):
    n_tiles = _n_patches(image_sizes_list, GRID_PINPOINTS, IMG_SIZE)
    n_img   = n_tiles * TOKENS_PER_PATCH
    raw     = text_tok.encode(text, add_special_tokens=True)
    img_pos = next((k for k,t in enumerate(raw) if t == IMAGE_TOKEN_ID), None)
    if img_pos is None:
        seq = text_tok.encode('<image>', add_special_tokens=False)
        for k in range(len(raw)-len(seq)+1):
            if raw[k:k+len(seq)] == seq:
                img_pos = k; raw = raw[:k]+[IMAGE_TOKEN_ID]+raw[k+len(seq):]; break
    if img_pos is None:
        pad = (raw + [text_tok.pad_token_id]*max_len)[:max_len]
        return pad, [1 if t != text_tok.pad_token_id else 0 for t in pad]
    expanded = raw[:img_pos] + [IMAGE_TOKEN_ID]*n_img + raw[img_pos+1:]
    if len(expanded) > max_len:
        avail    = max_len - n_img - img_pos
        expanded = raw[:img_pos] + [IMAGE_TOKEN_ID]*n_img + (raw[img_pos+1:][:avail] if avail>0 else [])
    pad = (expanded + [text_tok.pad_token_id]*max_len)[:max_len]
    return pad, [1 if t != text_tok.pad_token_id else 0 for t in pad]

def _mask_after_inst(ids):
    IE = text_tok.encode('[/INST]', add_special_tokens=False); labels = list(ids)
    for j in range(len(labels)-len(IE)+1):
        if labels[j:j+len(IE)] == IE:
            for k in range(j+len(IE)): labels[k] = -100
            break
    return labels

def _parse_bbox(raw):
    if isinstance(raw, (list, np.ndarray)): return [int(v) for v in raw]
    s = str(raw).replace('(','').replace(')','').replace('[','').replace(']','').strip()
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) != 4: parts = s.split()
    return [int(float(p)) for p in parts]

def free_memory():
    gc.collect()
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — STAGE I: PROJECTOR PRETRAINING
# ═══════════════════════════════════════════════════════════════════════════

s1_ckpt = None
s1_log  = []
s1_best = float('inf')

if SKIP_STAGE1:
    print("\nStage I SKIPPED.")
else:
    print("\n" + "=" * 60)
    print(" Step 4: Stage I — Projector Pretraining (VICReg)")
    print("=" * 60)
    free_memory()

    for p in model.parameters():
        p.requires_grad = False

    import bitsandbytes as bnb
    proj = (model.model.multi_modal_projector if hasattr(model, 'model')
            else model.multi_modal_projector)

    def dequant_module(module):
        for name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, bnb.nn.Params4bit):
                setattr(module, name, nn.Parameter(param.data.to(torch.float16), requires_grad=True))
            elif param.dtype not in (torch.float16, torch.bfloat16):
                setattr(module, name, nn.Parameter(param.data.cpu().to(torch.float16).to(param.device), requires_grad=True))
            else:
                param.requires_grad = True
        for child in module.children():
            dequant_module(child)

    dequant_module(proj)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_train:,} params (projector only)")
    assert n_train > 0, "No trainable params in projector"

    vision_tower = (model.model.vision_tower if hasattr(model, 'model') else model.vision_tower)
    projector    = proj

    def get_clip_features(pixel_values):
        B, n_tiles, C, H, W = pixel_values.shape
        pv_flat = pixel_values.view(B*n_tiles, C, H, W).to('cuda', dtype=torch.float16)
        with torch.no_grad():
            out  = vision_tower(pv_flat, output_hidden_states=True)
            feat = out.hidden_states[-2][:, 1:, :]
        return feat.detach()

    def get_proj_features(pixel_values):
        feat = get_clip_features(pixel_values)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return projector(feat)

    def get_proj_features_nograd(pixel_values):
        feat = get_clip_features(pixel_values)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return projector(feat)

    def align_loss(projected):
        fm   = projected.mean(dim=1).float()
        N, D = fm.shape
        if N < 2:
            return torch.tensor(0.0, device=fm.device, requires_grad=True)
        fm       = fm - fm.mean(dim=0, keepdim=True)
        std      = torch.sqrt(fm.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.mean(torch.relu(1.0 - std))
        cov      = (fm.T @ fm) / max(N - 1, 1)
        eye      = torch.eye(D, device=cov.device, dtype=cov.dtype)
        cov_loss = ((cov ** 2) * (1 - eye)).sum() / D
        return var_loss + 0.04 * cov_loss

    def run_val_loss(loader, max_batches=20):
        projector.eval(); losses = []
        for i, b in enumerate(loader):
            if i >= max_batches: break
            v = align_loss(get_proj_features_nograd(b['pixel_values'].to('cuda'))).item()
            if not (v != v) and abs(v) < 1e6: losses.append(v)
        projector.train()
        return float(np.mean(losses)) if losses else float('nan')

    class Stage1Dataset(Dataset):
        def __init__(self, df, cache=True):
            self.df    = df.drop_duplicates(subset='image_path').reset_index(drop=True)
            self.cache = {}
            print(f"Stage1Dataset: {len(self.df):,} images", end='')
            if cache:
                print(" - caching...", end='', flush=True)
                for idx, row in self.df.iterrows():
                    try:
                        img = Image.open(str(row['image_path'])).convert('RGB')
                        self.cache[idx] = img.resize((IMG_SIZE, IMG_SIZE))
                    except:
                        self.cache[idx] = Image.new('RGB',(IMG_SIZE,IMG_SIZE),(128,128,128))
                print(" done")
            else: print()
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            return {'image': self.cache.get(idx, Image.new('RGB',(IMG_SIZE,IMG_SIZE),(128,128,128)))}

    def s1_collate(batch):
        pv, sz = encode_images([b['image'] for b in batch])
        return {'pixel_values': pv, 'image_sizes': sz}

    uniq = train_df.drop_duplicates(subset='image_path')
    s1_df = uniq.reset_index(drop=True) if S1_SUBSAMPLE is None else \
            uniq.sample(min(len(uniq), S1_SUBSAMPLE), random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Stage I images: {len(s1_df):,}")
    val_uniq = val_df.drop_duplicates(subset='image_path').sample(
        min(200, len(val_df)), random_state=RANDOM_SEED).reset_index(drop=True)

    N_W      = min(8, os.cpu_count() or 4)
    DL_ARGS  = dict(num_workers=N_W, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    s1_tr_dl = DataLoader(Stage1Dataset(s1_df),    batch_size=8, shuffle=True,  collate_fn=s1_collate, **DL_ARGS)
    s1_vl_dl = DataLoader(Stage1Dataset(val_uniq), batch_size=8, shuffle=False, collate_fn=s1_collate, **DL_ARGS)
    print(f"Train: {len(s1_tr_dl):,} batches | {S1_EPOCHS} epochs")

    steps = (len(s1_tr_dl) // S1_GRAD_ACCUM) * S1_EPOCHS
    opt   = AdamW([p for p in model.parameters() if p.requires_grad], lr=S1_LR, weight_decay=1e-4)
    sched = get_linear_schedule_with_warmup(opt, int(steps*0.03), steps)

    for epoch in range(S1_EPOCHS):
        projector.train(); opt.zero_grad(); losses=[]; step=0
        pbar = tqdm(s1_tr_dl, desc=f"S1 E{epoch+1}/{S1_EPOCHS}")
        for i, batch in enumerate(pbar):
            pv   = batch['pixel_values'].to('cuda')
            feat = get_proj_features(pv)
            loss = align_loss(feat); v = loss.item()
            if v != v or abs(v) > 1e6:
                opt.zero_grad(); continue
            (loss / S1_GRAD_ACCUM).backward(); losses.append(v)
            if (i+1) % S1_GRAD_ACCUM == 0 or (i+1) == len(s1_tr_dl):
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                pbar.set_postfix({'loss': f'{np.mean(losses[-20:]):.4f}', 'gnorm': f'{float(gn):.2f}'})
        tr = float(np.mean(losses)) if losses else float('nan')
        vl = run_val_loss(s1_vl_dl)
        s1_log.append({'epoch': epoch+1, 'train': tr, 'val': vl})
        print(f"S1 E{epoch+1} — train={tr:.4f}  val={vl:.4f}")
        save_ok = not (vl != vl) and vl < s1_best
        if save_ok or s1_ckpt is None:
            if save_ok: s1_best = vl
            ckpt = CHECKPOINT_DIR / 'stage1_best'; ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(projector.state_dict(), ckpt / 'projector.pt')
            json.dump({'val_loss': vl if not (vl!=vl) else 999.0, 'epoch': epoch+1},
                      open(ckpt/'meta.json','w'))
            s1_ckpt = ckpt
            print(f"  -> {'best' if save_ok else 'saved (val=nan)'}: {ckpt}")

    del opt, sched; free_memory()
    for p in model.parameters(): p.requires_grad = True
    print(f"\nStage I done. Best val={s1_best:.4f}  Checkpoint: {s1_ckpt}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — STAGE II DATASETS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 5: Stage II — Datasets + DataLoaders")
print("=" * 60)

class Stage2Dataset(Dataset):
    def __init__(self, df):
        self.df = (df[df['question_type']!='conversation'].reset_index(drop=True)
                   if 'question_type' in df.columns else df.reset_index(drop=True))
        print(f"Stage2Dataset: {len(self.df):,} QA pairs")
    def __len__(self): return len(self.df)
    def _load(self, path):
        try: img = Image.open(str(path)); img.verify(); return Image.open(str(path)).convert('RGB')
        except: return Image.new('RGB',(IMG_SIZE,IMG_SIZE),(128,128,128))
    def _blend(self, img, bbox):
        arr = np.array(img); x1,y1,x2,y2 = bbox; ov = arr.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), (255,0,0), 3)
        alpha = random.uniform(96/255, 1.0) if ALPHA_MODE=='dynamic' else 200/255
        return Image.fromarray(cv2.addWeighted(ov,alpha,arr,1-alpha,0)).resize((IMG_SIZE,IMG_SIZE))
    def __getitem__(self, idx):
        row  = self.df.iloc[idx]; bbox = _parse_bbox(row['bbox'])
        return {'image':         self._blend(self._load(row['image_path']), bbox),
                'bbox_str':      row.get('bbox_str', f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"),
                'question':      str(row['question']),
                'answer':        str(row['answer']),
                'question_type': str(row.get('question_type','open'))}

def stage2_collate(batch):
    images = [b['image'] for b in batch]; pv, sz = encode_images(images)
    all_ids, all_masks, all_labels = [], [], []
    for b_i, b in enumerate(batch):
        text  = (f"[INST] <image>\nRegion of interest: {b['bbox_str']}\n"
                 f"{b['question']} [/INST] {b['answer']}")
        ids, mask = build_input_ids(text, sz[b_i].tolist())
        all_ids.append(ids); all_masks.append(mask); all_labels.append(_mask_after_inst(ids))
    return dict(pixel_values=pv, image_sizes=sz,
                input_ids=torch.tensor(all_ids, dtype=torch.long),
                attention_mask=torch.tensor(all_masks, dtype=torch.long),
                labels=torch.tensor(all_labels, dtype=torch.long))

DL_KW    = dict(num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
s2_train = Stage2Dataset(train_df)
s2_val   = Stage2Dataset(val_df)
s2_tr_dl = DataLoader(s2_train, batch_size=STAGE2_BATCH, shuffle=True,  collate_fn=stage2_collate, **DL_KW)
s2_vl_dl = DataLoader(s2_val,   batch_size=STAGE2_BATCH, shuffle=False, collate_fn=stage2_collate, **DL_KW)
print(f"S2: {len(s2_tr_dl):,} train | {len(s2_vl_dl):,} val batches")

b     = next(iter(s2_tr_dl))
n_img = (b['input_ids'] == IMAGE_TOKEN_ID).sum().item()
n_ans = (b['labels'] != -100).sum().item()
print(f"Sanity: image_tokens={n_img}, answer_tokens={n_ans}")
assert n_img > 0, "No image tokens — check TOKENS_PER_PATCH"
assert n_ans > 0, "No answer tokens — check collate"

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — STAGE II: LORA + TRAIN
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 6: Stage II — Apply LoRA + Train")
print("=" * 60)
free_memory()

from peft import get_peft_model, LoraConfig, TaskType

tgt = []
for name, _ in model.named_modules():
    if any(t in name for t in ["q_proj","k_proj","v_proj","o_proj"]):
        if any(s in name for s in ["language_model","llm","mistral","text_model"]):
            base = name.rsplit(".",1)[-1]
            if base not in tgt: tgt.append(base)
if not tgt:
    tgt = ["q_proj","k_proj","v_proj","o_proj"]
print(f"LoRA targets: {tgt}")

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=tgt, bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load Stage I projector
if s1_ckpt and (Path(str(s1_ckpt))/'projector.pt').exists():
    try:
        base_m = model.model if hasattr(model, 'model') else model
        pj     = (base_m.model.multi_modal_projector
                  if hasattr(base_m, 'model') else base_m.multi_modal_projector)
        pj.load_state_dict(torch.load(Path(str(s1_ckpt))/'projector.pt', map_location='cuda'))
        print(f"Stage I projector loaded from {s1_ckpt}")
    except Exception as e:
        print(f"Stage I projector load warning: {e}")
else:
    print("No Stage I projector — using pretrained LLaVA-v1.6 projector")

# Resume
s2_log, s2_best, s2_ckpt, patience, start_epoch = [], float('inf'), None, 0, 0
if RESUME_FROM and Path(RESUME_FROM).exists():
    from peft import PeftModel
    model = PeftModel.from_pretrained(model.model, str(RESUME_FROM))
    print(f"Resumed from: {RESUME_FROM}")
    meta_p = Path(RESUME_FROM) / 'meta.json'
    if meta_p.exists():
        meta        = json.load(open(meta_p))
        s2_best     = meta.get('val_loss', float('inf'))
        start_epoch = meta.get('epoch', 0)
        print(f"  epoch={start_epoch}  best_val={s2_best:.4f}")

model.train()

def val_loss_fn(model, loader, max_batches=30):
    model.eval(); losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            batch = {k: v.to('cuda', non_blocking=True) for k,v in batch.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(**{k:v for k,v in batch.items()
                               if k in ('pixel_values','image_sizes','input_ids','attention_mask','labels')})
            if out.loss is not None: losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float('nan')

def save_ckpt_s2(model, tag, v_loss, epoch=None):
    path = CHECKPOINT_DIR / tag; path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path); processor.save_pretrained(path)
    meta = {'val_loss': v_loss, 'tag': tag}
    if epoch is not None: meta['epoch'] = epoch
    json.dump(meta, open(path/'meta.json','w'))
    print(f"  Saved: {path}")
    return path

def train_epoch_s2(model, loader, optimizer, scheduler, grad_accum, label):
    model.train(); optimizer.zero_grad(); losses=[]; step=0
    pbar = tqdm(loader, desc=label)
    for i, batch in enumerate(pbar):
        batch = {k: v.to('cuda', non_blocking=True) for k,v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(**{k:v for k,v in batch.items()
                           if k in ('pixel_values','image_sizes','input_ids','attention_mask','labels')})
        loss = out.loss
        if loss is None: continue
        (loss / grad_accum).backward(); losses.append(loss.item())
        if (i+1) % grad_accum == 0 or (i+1) == len(loader):
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(); step += 1
            pbar.set_postfix({'loss': f'{np.mean(losses[-20:]):.4f}', 'gnorm': f'{float(gn):.2f}'})
    return float(np.mean(losses)) if losses else float('nan')

s2_steps = (len(s2_tr_dl) // STAGE2_GRAD_ACCUM) * STAGE2_EPOCHS
s2_opt   = AdamW([p for p in model.parameters() if p.requires_grad], lr=STAGE2_LR, weight_decay=0.)
s2_sched = get_linear_schedule_with_warmup(s2_opt, int(s2_steps * STAGE2_WARMUP), s2_steps)
print(f"Stage II: {STAGE2_EPOCHS} epochs | {s2_steps} steps | {len(s2_tr_dl):,} batches/epoch")

for epoch in range(start_epoch, STAGE2_EPOCHS):
    lbl = f"S2 Epoch {epoch+1}/{STAGE2_EPOCHS}"
    tr  = train_epoch_s2(model, s2_tr_dl, s2_opt, s2_sched, STAGE2_GRAD_ACCUM, lbl)
    vl  = val_loss_fn(model, s2_vl_dl)
    s2_log.append({'epoch': epoch+1, 'train': tr, 'val': vl, 'epoch_end': True})
    print(f"{lbl} — train={tr:.4f}  val={vl:.4f}")
    if vl < s2_best:
        s2_best = vl; patience = 0
        s2_ckpt = save_ckpt_s2(model, 'rllava_stage2_best', vl, epoch=epoch+1)
        print(f"  -> best (val={vl:.4f})")
    else:
        patience += 1
        print(f"  no improvement ({patience}/{EARLY_STOP_PATIENCE})")
        if patience >= EARLY_STOP_PATIENCE:
            print("Early stopping."); break
    save_ckpt_s2(model, 'rllava_stage2_last', vl, epoch=epoch+1)
    free_memory()

print(f"\nStage II done. Best val={s2_best:.4f}")
del s2_opt, s2_sched; free_memory()

# ─── Save training log + curve ───────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open(WORK_DIR/'training_log.json','w') as f:
    json.dump({'stage1': s1_log, 'stage2': s2_log,
               's1_best': s1_best, 's2_best': s2_best}, f, indent=2)
fig, axes = plt.subplots(1, 2, figsize=(14,5))
for ax, data, title in [
    (axes[0], s1_log, 'Stage I — Projector (VICReg)'),
    (axes[1], [e for e in s2_log if e.get('epoch_end')], 'Stage II — QLoRA')]:
    if not data: ax.set_title(title+' (skipped)'); continue
    ep = [e['epoch'] for e in data]
    ax.plot(ep, [e['train'] for e in data], 'o-',  label='Train', color='#2563eb')
    ax.plot(ep, [e['val']   for e in data], 's--', label='Val',   color='#f97316')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(WORK_DIR/'training_curve.png', dpi=150)
print(f"Training curve saved: {WORK_DIR/'training_curve.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 7: Evaluation — BLEU-4 / ROUGE-L / METEOR / EM")
print("=" * 60)

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as _meteor
from rouge_score import rouge_scorer as rs
for c in ['punkt','punkt_tab','wordnet']:
    try: nltk.download(c, quiet=True)
    except: pass

_rouge  = rs.RougeScorer(['rougeL'], use_stemmer=True)
_smooth = SmoothingFunction().method1

def bleu4(r,h):
    w = h.lower().split()
    return sentence_bleu([r.lower().split()], w, weights=(.25,)*4,
                         smoothing_function=_smooth) if w else 0.0
def rougel(r,h):  return _rouge.score(r,h)['rougeL'].fmeasure
def meteor(r,h):
    try: return _meteor([r.lower().split()], h.lower().split())
    except: return 0.0
def em(r,h): return float(r.strip().lower() == h.strip().lower())

print("Loading best checkpoint for evaluation...")
del model; free_memory()

from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

bnb_eval = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
_base = LlavaNextForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID, quantization_config=bnb_eval,
    device_map="auto", low_cpu_mem_usage=True)
ev = PeftModel.from_pretrained(_base, str(s2_ckpt))
ev.eval()

N_EVAL = min(500, len(s2_val)); preds = []
for idx in tqdm(range(N_EVAL), desc="Eval"):
    sample = s2_val[idx]
    prompt = (f"[INST] <image>\nRegion of interest: {sample['bbox_str']}\n"
              f"{sample['question']} [/INST]")
    inp = processor(text=prompt, images=sample['image'], return_tensors='pt')
    inp = {k: v.to('cuda') for k,v in inp.items()}
    with torch.no_grad():
        out = ev.generate(**inp, max_new_tokens=STAGE2_MAX_NEW_TOK, do_sample=False)
    pred = text_tok.decode(out[0, inp['input_ids'].shape[1]:],
                           skip_special_tokens=True).strip()
    preds.append({'pred': pred, 'gt': sample['answer'], 'qt': sample['question_type']})
    del inp, out
    if idx % 100 == 0: free_memory()
del ev; free_memory()

df_m = pd.DataFrame([{
    'qt': p['qt'],
    'b4': bleu4(p['gt'], p['pred']),
    'rl': rougel(p['gt'], p['pred']),
    'mt': meteor(p['gt'], p['pred']),
    'em': em(p['gt'], p['pred']),
} for p in preds])

print('\n' + '='*65)
print(f"{'Type':<22} {'BLEU-4':>8} {'ROUGE-L':>8} {'METEOR':>8} {'EM%':>7}  N")
print('-'*65)
for qt in df_m.qt.unique():
    s = df_m[df_m.qt==qt]
    print(f"  {qt:<20} {s.b4.mean():>8.4f} {s.rl.mean():>8.4f} "
          f"{s.mt.mean():>8.4f} {s.em.mean()*100:>6.1f}%  {len(s)}")
print('-'*65)
print(f"  {'OVERALL':<20} {df_m.b4.mean():>8.4f} {df_m.rl.mean():>8.4f} "
      f"{df_m.mt.mean():>8.4f} {df_m.em.mean()*100:>6.1f}%  {len(df_m)}")
print('='*65)

summary = {
    'n_eval': len(df_m), 'checkpoint': str(s2_ckpt),
    'overall': {
        'bleu4':       round(float(df_m.b4.mean()), 4),
        'rouge_l':     round(float(df_m.rl.mean()), 4),
        'meteor':      round(float(df_m.mt.mean()), 4),
        'exact_match': round(float(df_m.em.mean()), 4),
    },
    'by_question_type': {
        qt: {'bleu4':       round(float(df_m[df_m.qt==qt].b4.mean()), 4),
             'rouge_l':     round(float(df_m[df_m.qt==qt].rl.mean()), 4),
             'exact_match': round(float(df_m[df_m.qt==qt].em.mean()), 4),
             'n':           int((df_m.qt==qt).sum())}
        for qt in df_m.qt.unique()
    }
}
with open(WORK_DIR/'eval_metrics.json','w') as f: json.dump(summary, f, indent=2)
with open(WORK_DIR/'val_predictions.json','w') as f:
    json.dump([{'prediction': p['pred'], 'ground_truth': p['gt'],
                'question_type': p['qt']} for p in preds], f, indent=2)

print(f"\nAll outputs saved to {WORK_DIR}:")
print(f"  training_curve.png   training_log.json")
print(f"  eval_metrics.json    val_predictions.json")
print(f"  rllava_checkpoints/rllava_stage2_best/  <- final model")
