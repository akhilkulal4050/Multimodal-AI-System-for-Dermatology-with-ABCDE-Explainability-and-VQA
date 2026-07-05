#!/usr/bin/env python3
"""
NutriDermAI — RAG Corpus Builder
==================================
Builds ChromaDB vector index from dermatology corpus documents.

IMPORTANT — manual step required before running:
  Copy text from the URLs listed in Section 2 below into .txt files
  inside /data/Stagewise Dataset/Stage7/rag_corpus/<subfolder>/
  Then run this script to chunk + embed + index everything.

Usage:
    cd '/home/vjti-comp/Desktop/Final Project Code/VQA'
    python -u nutriderm_rag_build.py 2>&1 | tee rag_build.log

Set RESET=True to wipe and rebuild from scratch.
"""

import subprocess, sys, os, re, uuid
from pathlib import Path
from collections import Counter
from tqdm import tqdm

RESET = False

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — INSTALL PACKAGES
# ═══════════════════════════════════════════════════════════════════════════

def pip(*args):
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"] + list(args),
        capture_output=True, text=True)
    print(f"  [{'OK' if r.returncode==0 else 'FAIL'}] {args[0][:40]}")

print("=" * 60)
print(" Step 0: Installing packages")
print("=" * 60)
pip("chromadb>=0.4.0")
pip("sentence-transformers>=2.2.0")
pip("tqdm")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG + PATHS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 1: Config + Folder Setup")
print("=" * 60)

PROJECT_DIR    = Path('/home/vjti-comp/Desktop/Final Project Code/VQA')
RAG_CORPUS_DIR = Path('/data/Stagewise Dataset/Stage7/rag_corpus')
CHROMA_DB_DIR  = Path('/data/Stagewise Dataset/Stage7/chroma_db')

CHROMA_COLLECTION_NAME = 'dermatology_kb'
EMBEDDING_MODEL        = 'all-MiniLM-L6-v2'
CHUNK_SIZE             = 350
CHUNK_OVERLAP          = 60
MIN_CHUNK_CHARS        = 40
TOP_K                  = 3
SIMILARITY_THRESHOLD   = 0.25
EMBED_BATCH_SIZE       = 64

# Create folders
for sub in ['statpearls','abcde_guidelines','aad_guidelines','dermnet','medquad']:
    (RAG_CORPUS_DIR / sub).mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

print(f"  Corpus : {RAG_CORPUS_DIR}")
print(f"  ChromaDB: {CHROMA_DB_DIR}")
print(f"  Reset  : {RESET}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — WRITE HARDCODED QA PAIRS + CHECK MANUAL FILES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 2: Writing QA pairs + Checking manual corpus files")
print("=" * 60)

# ── 26 hardcoded dermatology QA pairs (no internet needed) ──────────────
DERM_QA = [
    ("What is melanoma?",
     "Melanoma is a malignant tumour of melanocytes, the pigment-producing cells of the skin. It is the most lethal form of skin cancer, responsible for over 57,000 deaths annually worldwide. Melanoma can arise from existing moles or appear as new pigmented lesions. Risk factors include ultraviolet exposure, fair skin, family history, and the presence of atypical nevi."),
    ("What are the ABCDE criteria for melanoma?",
     "The ABCDE criteria are: Asymmetry (one half does not mirror the other), Border irregularity (ragged, notched, or blurred edges), Color variation (multiple shades of brown, black, red, white, or blue-grey within the same lesion), Diameter greater than 6mm, and Evolution (any change in size, shape, color, or new symptoms such as bleeding or itching)."),
    ("What is the ABCD rule of dermoscopy?",
     "The dermoscopic ABCD rule scores lesions on four criteria: Asymmetry (0-2 points, evaluating two perpendicular axes), Border abruptness in 8 segments (0-8 points), Color richness counting up to 6 dermoscopic colors (1-6 points), and Dermoscopic structures including network, dots, streaks, and regression (1-5 points). A Total Dermoscopic Score above 5.45 is suspicious for melanoma."),
    ("What is the Total Dermoscopic Score?",
     "TDS = (Asymmetry score × 1.3) + (Border score × 0.1) + (Color score × 0.5) + (Dermoscopic structures score × 0.5). TDS below 4.75 suggests a benign lesion, 4.75 to 5.45 is borderline suspicious, and above 5.45 is highly suspicious for melanoma requiring excision."),
    ("What does asymmetry mean in dermoscopy?",
     "Asymmetry is evaluated by bisecting the lesion along two perpendicular axes through the geometric center. Score 0 means symmetric on both axes, score 1 means asymmetric on one axis, score 2 means asymmetric on both axes. An asymmetry score of 2 is a strong predictor of melanoma and contributes 2.6 points to the Total Dermoscopic Score."),
    ("What does border irregularity indicate in skin lesions?",
     "Irregular, poorly defined, or notched borders indicate potential malignancy. In the dermoscopic ABCD rule, the lesion border is divided into 8 pie-slice segments. Each segment showing an abrupt cutoff of pigment network scores 1 point. A high border score with ragged edges and satellite pigmentation is characteristic of melanoma."),
    ("What color variations indicate malignancy?",
     "The presence of 5 or 6 dermoscopic colors within a single lesion is strongly associated with melanoma. The six colors evaluated are: light brown, dark brown, black (indicating superficial or deep melanin), red (vascularity or inflammation), white (regression or fibrosis), and blue-grey (melanin in the dermis). Benign lesions typically show only 1-2 colors."),
    ("How is melanoma treated?",
     "Early melanoma is treated with wide local excision with margins of 0.5-2.0cm depending on Breslow thickness. Melanoma over 0.8mm requires sentinel lymph node biopsy. Advanced melanoma is treated with immunotherapy (pembrolizumab, nivolumab, ipilimumab), targeted therapy (vemurafenib, dabrafenib plus trametinib for BRAF V600 mutations), or radiation for brain metastases."),
    ("What surgical margins are used for melanoma?",
     "Recommended surgical margins by Breslow thickness: melanoma in situ requires 0.5-1.0cm margins; melanoma up to 1.0mm requires 1.0cm margins; 1.01-2.0mm requires 1-2cm margins; over 2.0mm requires 2.0cm margins. Sentinel lymph node biopsy is standard for melanomas greater than 0.8mm or those with ulceration or high mitotic rate."),
    ("What is the prognosis for melanoma by stage?",
     "Five-year survival rates by stage: Stage I (localized thin melanoma) over 98%, Stage II (localized thick or ulcerated) 65-90%, Stage III (regional lymph node involvement) 40-78%, Stage IV (distant metastasis) 15-20% with modern immunotherapy improving outcomes. Breslow thickness, ulceration, and mitotic rate are the most important prognostic factors."),
    ("What is basal cell carcinoma?",
     "Basal cell carcinoma is the most common skin cancer, arising from basal keratinocytes of the epidermis. It accounts for approximately 80% of non-melanoma skin cancers. BCC rarely metastasizes but can cause significant local tissue destruction. It most commonly appears on sun-exposed areas of the head and neck. Subtypes include nodular, superficial, morpheaform, and pigmented BCC."),
    ("How is basal cell carcinoma treated?",
     "BCC treatments include: surgical excision (cure rate 95% for primary BCC), Mohs micrographic surgery (98-99% cure rate for high-risk or recurrent BCC), electrodesiccation and curettage for small superficial lesions, cryotherapy, topical imiquimod or 5-fluorouracil for superficial BCC, and radiation therapy. Vismodegib or sonidegib (hedgehog pathway inhibitors) are used for locally advanced or metastatic BCC."),
    ("What is squamous cell carcinoma of the skin?",
     "Cutaneous squamous cell carcinoma arises from epidermal keratinocytes and is the second most common skin cancer. Unlike BCC, SCC has significant metastatic potential, particularly in high-risk locations (ear, lip, temple) and immunocompromised patients. SCC can arise from actinic keratoses, chronic wounds, or de novo. Staging follows the AJCC 8th edition system incorporating tumor size, depth, perineural invasion, and lymph node status."),
    ("How is squamous cell carcinoma treated?",
     "SCC treatment depends on risk category. Low-risk SCC is treated with excision (4-6mm margins) or curettage and electrodesiccation. High-risk SCC requires Mohs surgery or wide excision with 6-10mm margins, possible sentinel lymph node biopsy, and adjuvant radiation. Locally advanced or metastatic SCC is treated with cemiplimab (anti-PD-1) or pembrolizumab immunotherapy. Platinum-based chemotherapy is an alternative."),
    ("What is actinic keratosis?",
     "Actinic keratosis is a common premalignant epithelial lesion resulting from cumulative UV exposure. Clinically, AK presents as rough, scaly patches on sun-damaged skin. Histologically, AK shows atypical keratinocyte proliferation in the lower epidermis. The annual risk of individual AK progressing to invasive SCC is 0.1-0.5%, but patients with multiple AKs have substantially higher cumulative risk. Treatment includes cryotherapy, topical 5-fluorouracil, imiquimod, ingenol mebutate, photodynamic therapy, and diclofenac gel."),
    ("What is seborrheic keratosis?",
     "Seborrheic keratosis is the most common benign epithelial tumor, appearing as waxy, stuck-on plaques varying from light tan to dark brown or black. SK has no malignant potential and requires treatment only if irritated or cosmetically bothersome. Dermoscopic features pathognomonic for SK include comedo-like openings, milia-like cysts, fissures and ridges (brain-like pattern), and hairpin vessels. Treatment options include cryotherapy, curettage, electrodesiccation, and laser ablation."),
    ("How is seborrheic keratosis distinguished from melanoma?",
     "Key distinguishing features: SK has a stuck-on appearance with well-defined borders and warty surface texture, while melanoma has irregular borders and varied pigmentation. Dermoscopically, SK shows milia-like cysts, comedo-like openings, and sharp borders, whereas melanoma shows atypical pigment network, regression structures, and irregular vascular patterns. The ABCDE criteria are not typically positive for SK."),
    ("What is a melanocytic nevus?",
     "A melanocytic nevus (common mole) is a benign proliferation of melanocytes. Acquired nevi appear during the first three decades of life. Types include junctional (flat, at the dermal-epidermal junction), compound (slightly raised, both junctional and dermal components), and intradermal (raised, dome-shaped, predominantly dermal). Most nevi are stable and do not require treatment unless they show atypical features or the patient requests removal."),
    ("What is a dysplastic nevus?",
     "Dysplastic nevi (atypical moles, Clark nevi) are melanocytic nevi with clinical and histological atypia. They are typically larger than 5mm with irregular borders, variable pigmentation, and indistinct edges. Patients with multiple dysplastic nevi and family history of melanoma (Familial Atypical Multiple Mole Melanoma syndrome) have up to 500-fold increased lifetime melanoma risk. Annual full-body skin examination and regular dermoscopic monitoring are recommended."),
    ("When should a mole be removed?",
     "Mole removal is indicated when: ABCDE criteria are present, the lesion has changed rapidly, it bleeds or ulcerates spontaneously, dermoscopy reveals atypical features (irregular network, regression, atypical vessels), or the patient or clinician is uncertain. Any lesion with clinical or dermoscopic features suspicious for melanoma should be excised with 1-2mm margins for histopathological diagnosis rather than observed."),
    ("What is Mohs micrographic surgery?",
     "Mohs surgery is a specialized surgical technique for high-risk skin cancers where the tumor is removed in horizontal layers with immediate intraoperative microscopic margin assessment. Each layer is mapped, and removal continues only where tumor remains. This achieves the highest cure rates (98-99% for BCC, 92-97% for SCC) while conserving maximum normal tissue, making it ideal for tumors on the face, ears, eyelids, nose, and lips."),
    ("What is immunotherapy for skin cancer?",
     "Immune checkpoint inhibitors revolutionized advanced skin cancer treatment. Anti-PD-1 antibodies (pembrolizumab, nivolumab) and anti-CTLA-4 (ipilimumab) restore T-cell mediated tumor killing. Pembrolizumab is approved for advanced melanoma (first-line and adjuvant), advanced SCC, and Merkel cell carcinoma. Combination ipilimumab plus nivolumab achieves 5-year survival of approximately 52% in advanced melanoma. Immune-related adverse events can affect any organ system."),
    ("What factors increase risk of skin cancer?",
     "Major skin cancer risk factors: cumulative UV radiation exposure (sunlight, tanning beds), fair Fitzpatrick skin phototype I-II, personal or family history of skin cancer, multiple nevi (>50) or dysplastic nevi, immunosuppression (organ transplant recipients have 65-250× increased SCC risk), chronic scars or wounds, HPV infection (SCC), BRAF/NRAS mutations, Li-Fraumeni syndrome, xeroderma pigmentosum, and albinism."),
    ("What are high-risk features of SCC?",
     "High-risk SCC features per NCCN guidelines: diameter >2cm, depth >2mm or Clark level IV/V, poor differentiation, perineural or lymphovascular invasion, location on ear or non-hair-bearing lip, arising in scar or chronic inflammation, immunosuppressed patient, and recurrent tumor. These features increase metastatic risk to 10-30% and require more aggressive treatment including Mohs surgery, sentinel node biopsy consideration, and adjuvant radiation."),
    ("What is photodynamic therapy for skin lesions?",
     "Photodynamic therapy (PDT) uses a photosensitizing agent (aminolevulinic acid or methyl aminolevulinate) applied topically, which is preferentially absorbed by dysplastic cells. Subsequent activation with specific wavelength light (630nm red light) generates reactive oxygen species that selectively destroy abnormal cells. PDT is highly effective for actinic keratoses (clearance rate 70-90%), superficial BCC, and Bowen's disease. Advantages include excellent cosmetic outcomes and ability to treat large field cancerization areas."),
    ("How is dermoscopy used in clinical practice?",
     "Dermoscopy (dermatoscopy) is a non-invasive imaging technique using polarized or immersion light to visualize subsurface skin structures invisible to the naked eye. It improves melanoma detection sensitivity from 71% to 90% and specificity from 81% to 90% compared to naked-eye examination. Key dermoscopic structures evaluated include: pigment network, aggregated globules, streaks/pseudopods, regression structures, milia-like cysts, and vascular patterns. Pattern analysis and algorithmic methods (ABCD rule, 7-point checklist, Menzies method) guide clinical decisions."),
]

out_dir = RAG_CORPUS_DIR / 'medquad'
written = 0
for i, (q, a) in enumerate(DERM_QA):
    fname = f"derm_qa_{i+1:03d}.txt"
    (out_dir / fname).write_text(f"Q: {q}\nA: {a}\n", encoding='utf-8')
    written += 1
print(f"  [OK] {written} hardcoded QA pairs written to medquad/")

# ── Check which manual files are present ────────────────────────────────
print("\n  Manual corpus files check:")
print("  (Open these URLs in browser, select all text, paste into the .txt files)\n")

manual_files = {
    'statpearls': [
        ('melanoma.txt',              'https://www.ncbi.nlm.nih.gov/books/NBK519567/'),
        ('basal_cell_carcinoma.txt',  'https://www.ncbi.nlm.nih.gov/books/NBK482439/'),
        ('squamous_cell.txt',         'https://www.ncbi.nlm.nih.gov/books/NBK441939/'),
        ('actinic_keratosis.txt',     'https://www.ncbi.nlm.nih.gov/books/NBK557401/'),
        ('seborrheic_keratosis.txt',  'https://www.ncbi.nlm.nih.gov/books/NBK545285/'),
        ('melanocytic_nevus.txt',     'https://www.ncbi.nlm.nih.gov/books/NBK459260/'),
    ],
    'abcde_guidelines': [
        ('aad_abcde.txt',             'https://www.aad.org/public/diseases/skin-cancer/find/at-risk/abcdes'),
        ('dermnet_abcde.txt',         'https://dermnetnz.org/topics/dermoscopy-of-melanoma'),
    ],
    'aad_guidelines': [
        ('melanoma.txt',              'https://www.aad.org/public/diseases/skin-cancer/types/common/melanoma'),
        ('bcc.txt',                   'https://www.aad.org/public/diseases/skin-cancer/types/common/bcc'),
        ('scc.txt',                   'https://www.aad.org/public/diseases/skin-cancer/types/common/scc'),
        ('actinic_keratosis.txt',     'https://www.aad.org/public/diseases/a-z/actinic-keratosis-overview'),
    ],
    'dermnet': [
        ('melanoma.txt',              'https://dermnetnz.org/topics/melanoma'),
        ('bcc.txt',                   'https://dermnetnz.org/topics/basal-cell-carcinoma'),
        ('scc.txt',                   'https://dermnetnz.org/topics/squamous-cell-carcinoma'),
        ('actinic_keratosis.txt',     'https://dermnetnz.org/topics/actinic-keratosis'),
        ('seborrhoeic_keratosis.txt', 'https://dermnetnz.org/topics/seborrhoeic-keratosis'),
        ('melanocytic_naevus.txt',    'https://dermnetnz.org/topics/melanocytic-naevus'),
    ],
}

missing_count = 0
found_count   = 0
for folder, files in manual_files.items():
    folder_path = RAG_CORPUS_DIR / folder
    for fname, url in files:
        fpath  = folder_path / fname
        exists = fpath.exists() and fpath.stat().st_size > 200
        tag    = "OK  " if exists else "MISS"
        if exists:
            found_count += 1
            print(f"    [{tag}] {folder}/{fname}  ({fpath.stat().st_size//1024}KB)")
        else:
            missing_count += 1
            print(f"    [{tag}] {folder}/{fname}")
            print(f"           -> {url}")

print(f"\n  Manual files: {found_count} found, {missing_count} missing")
if missing_count > 0:
    print(f"\n  NOTE: {missing_count} files missing — RAG will still work using")
    print(f"  the {written} hardcoded QA pairs. Add manual files to improve quality.")
    print(f"\n  Minimum recommended: paste statpearls/ files (6 URLs above)")
    print(f"  They give the most clinical detail for the 6 disease classes.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CHUNK
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 3: Chunking documents")
print("=" * 60)

def chunk_text(text, source='', doc_id=''):
    text  = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if not words: return []
    chunks=[]; step=max(1, CHUNK_SIZE-CHUNK_OVERLAP)
    for i, start in enumerate(range(0, len(words), step)):
        ct = ' '.join(words[start: start+CHUNK_SIZE])
        if len(ct) >= MIN_CHUNK_CHARS:
            chunks.append({'text': ct, 'source': source,
                           'doc_id': doc_id, 'chunk_index': i})
    return chunks

docs = [f for f in RAG_CORPUS_DIR.rglob('*.txt') if f.is_file()]
print(f"  Found {len(docs)} documents")
all_chunks = []
for path in sorted(docs):
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
        all_chunks.extend(chunk_text(text, source=path.parent.name, doc_id=path.stem))
    except Exception as e:
        print(f"  Warning: {path.name}: {e}")

src_counts = Counter(c['source'] for c in all_chunks)
print(f"  Generated {len(all_chunks):,} chunks:")
for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
    print(f"    {src:<25} {cnt:>5,} chunks")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — EMBED + INDEX
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 4: Embedding + Indexing into ChromaDB")
print("=" * 60)

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

print(f"  Loading {EMBEDDING_MODEL} ...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
print("  Model loaded.")

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR),
    settings=Settings(anonymized_telemetry=False),
)

if RESET:
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"  Deleted existing collection.")
    except Exception: pass

collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

existing = collection.count()
if existing > 0 and not RESET:
    print(f"  Collection already has {existing:,} chunks.")
    print(f"  Set RESET=True at top of script to rebuild.")
else:
    print(f"  Indexing {len(all_chunks):,} chunks...")
    for start in tqdm(range(0, len(all_chunks), EMBED_BATCH_SIZE), desc="  Embedding"):
        batch  = all_chunks[start: start+EMBED_BATCH_SIZE]
        texts  = [c['text'] for c in batch]
        embeds = embed_model.encode(texts, normalize_embeddings=True).tolist()
        ids    = [f"{c['doc_id']}__{c['chunk_index']}__{uuid.uuid4().hex[:6]}"
                  for c in batch]
        metas  = [{'source': c['source'], 'doc_id': c['doc_id'],
                   'chunk_index': c['chunk_index']} for c in batch]
        collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
    print(f"  Indexed {collection.count():,} chunks.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — TEST RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" Step 5: Test Retrieval")
print("=" * 60)

def retrieve(question, top_k=3, source_filter=None):
    emb   = embed_model.encode([question], normalize_embeddings=True).tolist()
    where = {'source': source_filter} if source_filter else None
    res   = collection.query(query_embeddings=emb, n_results=top_k*2, where=where)
    out   = []
    for doc, meta, dist in zip(res['documents'][0],
                                res['metadatas'][0],
                                res['distances'][0]):
        sim = 1.0 - dist
        if sim >= SIMILARITY_THRESHOLD:
            out.append({'text': doc, 'source': meta['source'], 'score': round(sim,4)})
        if len(out) >= top_k: break
    return out

tests = [
    ("What are the ABCDE criteria for melanoma?",     None),
    ("How is basal cell carcinoma treated?",          None),
    ("What causes actinic keratosis?",                None),
    ("What is the Total Dermoscopic Score?",          "medquad"),
    ("How is squamous cell carcinoma diagnosed?",     None),
    ("What is a seborrheic keratosis?",               None),
    ("When should a mole be removed?",                "medquad"),
    ("What surgical margins are used for melanoma?",  None),
]

hits = 0
for q, filt in tests:
    results = retrieve(q, top_k=2, source_filter=filt)
    tag     = "OK" if results else "MISS"
    if results: hits += 1
    print(f"  [{tag}] {q}")
    for r in results:
        print(f"        [{r['score']:.3f}] [{r['source']}] {r['text'][:90]}...")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" RAG Index Complete")
print("=" * 60)
print(f"  Chunks in ChromaDB : {collection.count():,}")
print(f"  Test queries       : {hits}/{len(tests)} returned results")
print(f"  Corpus location    : {RAG_CORPUS_DIR}")
print(f"  ChromaDB location  : {CHROMA_DB_DIR}")
print()
if missing_count > 0:
    print(f"  To improve RAG quality, add {missing_count} missing corpus files.")
    print(f"  Copy text from the URLs shown in Step 2 into the .txt files.")
    print(f"  Then re-run with RESET=True to rebuild the index.")
    print()
print(f"  Start the bot:")
print(f"    cd '{PROJECT_DIR}'")
print(f"    python dermatology_bot.py")
print("=" * 60)
