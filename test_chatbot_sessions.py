#!/usr/bin/env python3
"""
NutriDermAI Chatbot — Session Test Suite
=========================================
Tests 3 patient sessions covering all major flows:
  Session 1: High-risk melanoma → diagnosis + ABCDE + risk + treatment Qs
  Session 2: Benign nevus      → reassurance + monitoring Qs
  Session 3: Resume from disk  → resume existing session without re-uploading image

Run:
    python test_chatbot_sessions.py

Requires:
  - dermatology_bot.py server running on localhost:8000
    OR run in standalone mode (set STANDALONE=True below)
"""

import sys, json, time, requests
from pathlib import Path
from PIL import Image
import numpy as np
import io

# ── Config ────────────────────────────────────────────────────────────────────
BOT_URL    = "http://localhost:8000"
STANDALONE = False   # True = test without server (import bot directly)
VQA_DIR    = Path("/home/vjti-comp/Desktop/Final Project Code/VQA")

# ── Colour output ─────────────────────────────────────────────────────────────
G  = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; E = "\033[0m"

def ok(msg):   print(f"{G}  ✓ {msg}{E}")
def fail(msg): print(f"{R}  ✗ {msg}{E}"); 
def info(msg): print(f"{B}  → {msg}{E}")
def head(msg): print(f"\n{Y}{'='*60}{E}\n{Y}  {msg}{E}\n{Y}{'='*60}{E}")


# ── Synthetic test images (no real images needed) ────────────────────────────
def make_test_image(pattern: str = 'melanoma') -> bytes:
    """Create a simple synthetic skin lesion image for testing."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 220  # skin-coloured background
    if pattern == 'melanoma':
        # Asymmetric dark irregular patch
        import cv2
        pts = np.array([[200,150],[320,140],[360,260],[280,320],[180,300],[160,200]], np.int32)
        cv2.fillPoly(img, [pts], (40, 25, 15))       # dark brown/black
        cv2.fillPoly(img, [pts+20], (80, 50, 30))    # lighter ring
        cv2.circle(img, (310, 190), 20, (20, 10, 5), -1)  # satellite
    elif pattern == 'nevus':
        # Symmetric round brown mole
        import cv2
        cv2.circle(img, (256, 256), 60, (100, 65, 40), -1)
        cv2.circle(img, (256, 256), 58, (120, 80, 55), -1)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG')
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# HTTP MODE — test via running FastAPI server
# ══════════════════════════════════════════════════════════════════════════════

def http_new_session() -> str:
    r = requests.post(f"{BOT_URL}/session/new", timeout=10)
    r.raise_for_status()
    sid = r.json()['session_id']
    ok(f"New session created: {sid}")
    return sid

def http_upload_image(session_id: str, image_bytes: bytes,
                       disease_label=None, risk_score=None,
                       A=None, B=None, C=None, D=None, E=None) -> dict:
    files  = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
    data   = {}
    if disease_label: data['disease_label'] = disease_label
    if risk_score:    data['risk_score']    = str(risk_score)
    if A is not None: data['A'] = str(A)
    if B is not None: data['B'] = str(B)
    if C is not None: data['C'] = str(C)
    if D is not None: data['D'] = str(D)
    if E is not None: data['E'] = str(E)
    r = requests.post(f"{BOT_URL}/session/{session_id}/image",
                      files=files, data=data, timeout=60)
    r.raise_for_status()
    result = r.json()
    ok(f"Image processed → bbox: {result.get('bbox')}")
    if result.get('risk_tier'):
        info(f"Risk tier from Stage 6: {result['risk_tier']}")
    return result

def http_ask(session_id: str, question: str) -> str:
    r = requests.post(f"{BOT_URL}/session/{session_id}/ask",
                      data={'question': question}, timeout=120)
    r.raise_for_status()
    result = r.json()
    answer = result['answer']
    ok(f"Turn {result['turn_count']} answered")
    print(f"     Q: {question}")
    print(f"     A: {answer[:200]}{'...' if len(answer)>200 else ''}")
    return answer

def http_history(session_id: str) -> list:
    r = requests.get(f"{BOT_URL}/session/{session_id}/history", timeout=10)
    r.raise_for_status()
    return r.json()['turns']

def http_end_session(session_id: str):
    r = requests.delete(f"{BOT_URL}/session/{session_id}", timeout=10)
    ok(f"Session {session_id} closed")

def http_resume_session(session_id: str, question: str) -> str:
    """Ask a question in a previously saved session — no image upload needed."""
    r = requests.post(f"{BOT_URL}/session/{session_id}/ask",
                      data={'question': question}, timeout=120)
    r.raise_for_status()
    return r.json()['answer']


# ══════════════════════════════════════════════════════════════════════════════
# TEST SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

def test_session_1_melanoma():
    """
    Session 1: High-risk melanoma
    Simulates Stage 6 passing disease_label='MEL' and ABCDE scores.
    Tests: diagnosis Q, ABCDE Q, risk Q, treatment Q, differential Q.
    Expected: answers should reflect HIGH/URGENT risk and excision treatment.
    """
    head("SESSION 1: High-risk melanoma (with Stage 6 clinical context)")
    sid = http_new_session()

    # Upload image + pass Stage 6 clinical context
    img = make_test_image('melanoma')
    http_upload_image(
        sid, img,
        disease_label='MEL',
        risk_score=0.88,
        A=0.87, B=0.74, C=0.62, D=0.71, E=0.82,  # all HIGH
    )

    questions = [
        ("diagnosis",    "What skin condition is shown in this image?"),
        ("abcde",        "What are the ABCDE features of this lesion?"),
        ("risk",         "How serious is this lesion? Is it dangerous?"),
        ("treatment",    "What treatment is recommended for this patient?"),
        ("differential", "Could this be a benign nevus instead of melanoma?"),
        ("followup",     "What should the patient do next?"),
    ]

    results = {}
    for category, q in questions:
        info(f"Testing {category} question...")
        answer = http_ask(sid, q)
        results[category] = answer

    # Basic assertions
    diagnosis_ans = results['diagnosis'].lower()
    assert any(w in diagnosis_ans for w in ['melanoma','malignant','concern','atypical']), \
        f"Diagnosis answer doesn't mention melanoma: {diagnosis_ans[:100]}"
    ok("Diagnosis answer mentions melanoma/malignancy ✓")

    treatment_ans = results['treatment'].lower()
    assert any(w in treatment_ans for w in ['excision','surgery','biopsy','refer','dermatol']), \
        f"Treatment answer doesn't mention surgery: {treatment_ans[:100]}"
    ok("Treatment answer mentions surgical intervention ✓")

    # Check history
    history = http_history(sid)
    assert len(history) == len(questions) * 2  # user + assistant per turn
    ok(f"History has {len(history)} turns ({len(questions)} exchanges) ✓")

    http_end_session(sid)
    print(f"\n  Session 1 PASSED ✓  session_id={sid}")
    return sid   # return for Session 3 resume test


def test_session_2_nevus():
    """
    Session 2: Benign nevus
    No Stage 6 clinical context passed — tests fallback to R-LLaVA only.
    Tests: diagnosis Q, reassurance Q, monitoring Q.
    Expected: reassuring answers, no urgent referral language.
    """
    head("SESSION 2: Benign nevus (no Stage 6 context — R-LLaVA only)")
    sid = http_new_session()

    img = make_test_image('nevus')
    http_upload_image(sid, img)  # no clinical context

    questions = [
        ("diagnosis",  "What is this skin lesion?"),
        ("benign",     "Is this mole dangerous?"),
        ("monitoring", "How often should I get this checked?"),
    ]

    results = {}
    for category, q in questions:
        info(f"Testing {category} question...")
        answer = http_ask(sid, q)
        results[category] = answer

    http_end_session(sid)
    print(f"\n  Session 2 PASSED ✓  session_id={sid}")
    return sid


def test_session_3_resume(session_1_id: str):
    """
    Session 3: Resume a closed session from disk.
    No new image upload — asks a follow-up question in Session 1's context.
    Tests: ConversationMemory.load(), session resume, history persistence.
    """
    head("SESSION 3: Resume Session 1 from disk (no image re-upload)")
    info(f"Resuming session: {session_1_id}")

    # Session was closed in test 1 but saved to disk — resume it
    history_before = http_history(session_1_id)
    info(f"History loaded: {len(history_before)} turns from disk")
    assert len(history_before) > 0, "Session history should be non-empty after resume"
    ok(f"Session resumed from disk with {len(history_before)} turns ✓")

    # Ask a follow-up — no image upload, uses stored bbox + blended image
    followup = http_resume_session(
        session_1_id,
        "Based on what we discussed, when exactly should the patient see a doctor?"
    )
    print(f"     Follow-up A: {followup[:200]}")
    ok("Follow-up answered using existing session context ✓")

    history_after = http_history(session_1_id)
    assert len(history_after) > len(history_before), "History should have grown"
    ok(f"History grew: {len(history_before)} → {len(history_after)} turns ✓")

    print(f"\n  Session 3 PASSED ✓")


def test_health():
    """Test server health and session listing."""
    head("PRE-CHECK: Server health")
    r = requests.get(f"{BOT_URL}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    ok(f"Server healthy: {data}")
    if data.get('vram_gb', 0) < 1:
        print(f"{Y}  ⚠ VRAM < 1GB — models may not be loaded{E}")
    else:
        ok(f"VRAM in use: {data['vram_gb']} GB — models are loaded")


# ══════════════════════════════════════════════════════════════════════════════
# HOW TO EXECUTE THE CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

EXECUTION_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║        HOW TO EXECUTE THE NutriDermAI CHATBOT               ║
╚══════════════════════════════════════════════════════════════╝

STEP 1 — Copy R-LLaVA checkpoint from DGX to RTX 4070 machine:
──────────────────────────────────────────────────────────────
  scp -r prasannam24-26@172.18.33.4:/home/prasannam24-26/rllava/checkpoints/rllava_stage2_best \\
      "/home/vjti-comp/Desktop/Final Project Code/VQA/rllava/"

  Expected structure after copy:
  VQA/
  └── rllava/
      └── rllava_stage2_best/          ← LoRA adapters
          ├── adapter_config.json
          ├── adapter_model.safetensors
          └── meta.json

STEP 2 — Verify ChromaDB is built (RAG index):
──────────────────────────────────────────────
  python3 -c "
  import chromadb
  from chromadb.config import Settings
  c = chromadb.PersistentClient(
      path='/home/vjti-comp/Desktop/Final Project Code/VQA/chroma_db',
      settings=Settings(anonymized_telemetry=False))
  col = c.get_collection('dermatology_kb')
  print(f'RAG index: {col.count()} chunks ready')
  "
  If 0 chunks → run dgx_rag_pipeline.ipynb Cell 6 first.

STEP 3 — Install dependencies on RTX 4070:
────────────────────────────────────────────
  pip install fastapi uvicorn transformers peft bitsandbytes \\
              accelerate chromadb sentence-transformers pillow \\
              opencv-python-headless ollama

STEP 4 — (Optional) Start Ollama for Llama reasoning:
───────────────────────────────────────────────────────
  ollama serve &
  ollama pull llama3.1:8b
  # If skipped, Llama reasoning step is auto-disabled gracefully

STEP 5 — Start the FastAPI server:
────────────────────────────────────
  cd "/home/vjti-comp/Desktop/Final Project Code/VQA"
  python dermatology_bot.py

  # OR with uvicorn:
  uvicorn dermatology_bot:app --host 0.0.0.0 --port 8000

  # Background:
  nohup python -u dermatology_bot.py > bot.log 2>&1 &
  tail -f bot.log

STEP 6 — Open the web interface:
──────────────────────────────────
  Open index.html in your browser:
  file:///home/vjti-comp/Desktop/Final Project Code/VQA/index.html

  OR if served via Python:
  python3 -m http.server 3000
  → http://localhost:3000/index.html

STEP 7 — Run these tests:
───────────────────────────
  python test_chatbot_sessions.py

SESSIONS NEEDED:
────────────────
  • 1 session per patient visit (image upload = new session)
  • Session is automatically saved to VQA/sessions/{id}.json after each turn
  • For a full test run you need 3 sessions (see test cases above)
  • For a real patient demo: 1 session per image, unlimited follow-up Qs

API QUICK REFERENCE:
─────────────────────
  POST /session/new                    → create session, get session_id
  POST /session/{id}/image             → upload image (ARCUNet+SLRC runs)
  POST /session/{id}/ask   ?question=  → ask a question
  GET  /session/{id}/history           → get full conversation
  GET  /session/{id}/bbox              → get lesion bounding box
  DELETE /session/{id}                 → close and save session
  GET  /health                         → server health + VRAM

STAGE 6 CLINICAL CONTEXT (optional — pass with image upload):
───────────────────────────────────────────────────────────────
  POST /session/{id}/image
    disease_label=MEL    (MEL/BCC/SCC/ACK/NEV/SEK)
    risk_score=0.88
    A=0.87  B=0.74  C=0.62  D=0.71  E=0.82
  → Enables rules engine from first question

"""

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print(EXECUTION_GUIDE)

    if '--guide-only' in sys.argv:
        sys.exit(0)

    print("\n" + "="*60)
    print("  Running Session Tests")
    print("="*60)

    # Check server is up
    try:
        test_health()
    except Exception as e:
        print(f"{R}Server not reachable at {BOT_URL}: {e}{E}")
        print(f"{Y}Start the server first: python dermatology_bot.py{E}")
        print(f"{Y}Then re-run: python test_chatbot_sessions.py{E}")
        sys.exit(1)

    passed = 0; failed_tests = []

    # Session 1
    try:
        s1_id = test_session_1_melanoma()
        passed += 1
    except Exception as e:
        print(f"{R}Session 1 FAILED: {e}{E}")
        failed_tests.append(f"Session 1: {e}")
        s1_id = None

    # Session 2
    try:
        test_session_2_nevus()
        passed += 1
    except Exception as e:
        print(f"{R}Session 2 FAILED: {e}{E}")
        failed_tests.append(f"Session 2: {e}")

    # Session 3 (only if Session 1 succeeded)
    if s1_id:
        try:
            test_session_3_resume(s1_id)
            passed += 1
        except Exception as e:
            print(f"{R}Session 3 FAILED: {e}{E}")
            failed_tests.append(f"Session 3: {e}")

    # Summary
    head(f"RESULTS: {passed}/3 tests passed")
    if failed_tests:
        for f in failed_tests:
            print(f"{R}  ✗ {f}{E}")
    else:
        print(f"{G}  All tests passed! Chatbot is working correctly.{E}")
