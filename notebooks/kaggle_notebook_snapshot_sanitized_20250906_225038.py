# Kaggle notebook code snapshot (sanitized)

# ==== Cell 1 ====
# Use %pip so installs affect THIS kernel
get_ipython().run_line_magic('pip', '-q install -U pip setuptools wheel')

# 0) Clean out things that often cause conflicts
get_ipython().run_line_magic('pip', '-q uninstall -y torch torchvision torchaudio preprocessing google-cloud-automl || true')

# 1) PyTorch for CUDA 12.1 (Kaggle GPUs)
get_ipython().run_line_magic('pip', '-q install --no-cache-dir --force-reinstall    --index-url https://download.pytorch.org/whl/cu121    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1')

# 2) Scientific stack (versions compiled for NumPy 1.26 ABI)
get_ipython().run_line_magic('pip', '-q install --no-cache-dir --force-reinstall    numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2')

# 3) Core NLP libs
get_ipython().run_line_magic('pip', '-q install    "transformers>=4.43,<5"    "datasets>=2.19,<3"    "evaluate>=0.4.2,<0.5"    conllu>=4.5.3 sentencepiece>=0.1.99    tqdm pandas nltk==3.9.1 supar==1.1.4')

# 4) Harmony pins
# - Protobuf 4.25.3 prevents the MessageFactory/GetPrototype error but stays compatible with HF
# - datasets 2.21.0 wants fsspec <= 2024.6.1 (align gcsfs too)
get_ipython().run_line_magic('pip', '-q install --no-cache-dir --force-reinstall protobuf==4.25.3 fsspec==2024.6.1 gcsfs==2024.6.1')

# Optional: quieter logs; avoids a lot of TF/XLA noise if TF is present in the image
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Optional: purge stale wheels
get_ipython().run_line_magic('pip', 'cache purge')

print("✅ Setup complete. Now restart the runtime (Kernel/Runtime → Restart) before running the next cell.")

# ==== Cell 2 ====
import os
# Keep TF/XLA chatty logs down if TF happens to be preinstalled
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch, supar, transformers, datasets, evaluate, conllu, nltk
import importlib.metadata

print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("supar:", supar.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("evaluate:", evaluate.__version__)
print("conllu:", importlib.metadata.version("conllu"))
print("nltk:", nltk.__version__)
print("OK ✅")

# ==== Cell 3 ====
import requests
from pathlib import Path
from conllu import parse_incr

# --- Clean, correct config ---
UD_RELEASE = "2.16"
UD_REPOS = {
    "en_ewt":  "UniversalDependencies/UD_English-EWT",
    "hi_hdtb": "UniversalDependencies/UD_Hindi-HDTB",
    "ur_udtb": "UniversalDependencies/UD_Urdu-UDTB",
    "bn_bru":  "UniversalDependencies/UD_Bengali-BRU",
    "bn_pud":  "UniversalDependencies/UD_Bengali-PUD",
}

# What splits are known to exist for r2.16 (avoid 404 spam)
AVAILABLE_SPLITS = {
    "en_ewt":  ["train", "dev", "test"],
    "hi_hdtb": ["train", "dev", "test"],
    "ur_udtb": ["train", "dev", "test"],
    "bn_bru":  ["test"],               # BRU only has test
    "bn_pud":  [],                     # PUD has no files in r2.16 for Bengali
}

def ud_url(tb: str, split: str, release: str = UD_RELEASE) -> str:
    # Produce a raw.githubusercontent.com URL to the file on the rX.YY tag
    return f"https://raw.githubusercontent.com/{UD_REPOS[tb]}/r{release}/{tb}-ud-{split}.conllu"

DATA = Path("./data/ud"); DATA.mkdir(parents=True, exist_ok=True)

def fetch(tb: str, sp: str):
    """Download a UD file if it exists; return Path or None."""
    p = DATA / tb / f"{tb}-{sp}.conllu"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        print("CACHED", p)
        return p
    url = ud_url(tb, sp)
    try:
        r = requests.get(url, timeout=60)
    except Exception as e:
        print("MISS ", url, "ERR", e)
        return None
    if r.status_code != 200:
        print("MISS ", url, r.status_code)
        return None
    p.write_bytes(r.content)
    print("OK   ", p)
    return p

def ensure_all():
    files = []
    for tb, splits in AVAILABLE_SPLITS.items():
        for sp in splits:
            p = fetch(tb, sp)
            if p is not None:
                files.append(p)
    return files

def read_conllu_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            toks   = [t["form"]   for t in sent if isinstance(t["id"], int)]
            upos   = [t["upos"]   for t in sent if isinstance(t["id"], int)]
            head   = [t["head"]   for t in sent if isinstance(t["id"], int)]
            deprel = [t["deprel"] for t in sent if isinstance(t["id"], int)]
            rows.append({"tokens": toks, "upos": upos, "head": head, "deprel": deprel})
    return rows

files = ensure_all()
print("\nAvailable files:")
for f in files:
    print(" -", f)

# ==== Cell 4 ====
# Zero-shot BN -> Train EN (EWT) -> EN dev eval -> BN re-test + comparisons
# - Saves logs, learning_curve.csv, metrics.json, predictions
# - Robust to SuPar/transformers quirks (no parser=train() reassign, JSON-safe metrics)
# ------------------------------------------------------------------------------------------------
import os, io, sys, json, re
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# ---- BEFORE importing torch/transformers: CUDA & logging knobs ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb=128")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# HF optimizer shim (SuPar expects these names)
# ---------------------------
try:
    from transformers import AdamW as _TestAdamW  # noqa: F401
    from transformers import get_linear_schedule_with_warmup as _TestSched  # noqa: F401
except Exception:
    import transformers
    import torch.optim as optim
    from torch.optim.lr_scheduler import LambdaLR

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
        return LambdaLR(optimizer, lr_lambda)

    transformers.AdamW = optim.AdamW
    transformers.get_linear_schedule_with_warmup = get_linear_schedule_with_warmup
# ---------------------------

from supar.parsers import CRF2oDependencyParser

# Make matmul cheaper on T4
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ---- config/paths ----
XLMR = "xlm-roberta-base"
DATA = Path("./data/ud")

# EN (EWT)
EN_DIR = DATA / "en_ewt"
EN_TRAIN = EN_DIR / "en_ewt-train.conllu"
EN_DEV   = EN_DIR / "en_ewt-dev.conllu"

# BN (BRU) — test only for now
BN_DIR   = DATA / "bn_bru"
BN_TEST  = BN_DIR / "bn_bru-test.conllu"

# Outputs
OUT = Path("./outputs_dep"); OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUT / "supar_crf2o_xlmr_en"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PT = MODEL_DIR / "parser.pt"
LOG_TXT  = MODEL_DIR / "train_stdout.txt"
CURVE_CSV= MODEL_DIR / "learning_curve.csv"
METRICS_JSON = MODEL_DIR / "metrics.json"

# BN predictions (pre vs post training)
BN_PRED_PRE  = MODEL_DIR / "bn_bru-test.pretrain.pred.conllu"  # zero-shot
BN_PRED_POST = MODEL_DIR / "bn_bru-test.posttrain.pred.conllu" # after EN training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---- sanity checks ----
assert EN_TRAIN.exists(), f"Missing {EN_TRAIN}"
assert EN_DEV.exists(),   f"Missing {EN_DEV}"
assert BN_TEST.exists(),  f"Missing {BN_TEST} (required for BN evaluation)"

# ---- log tee ----
class Tee(io.StringIO):
    """
    Writes to: 1) live console, 2) a file, and 3) in-memory buffer (self).
    You can choose file mode: 'w' (overwrite) or 'a' (append).
    """
    def __init__(self, file_path: Path, mode: str = "w"):
        super().__init__()
        self._f = open(file_path, mode, encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self._f.write(s)
        return super().write(s)

    def flush(self):
        sys.__stdout__.flush()
        self._f.flush()
        return super().flush()

    def close(self):
        try:
            self._f.close()
        finally:
            super().close()

# ---- JSON-safe metrics helpers ----
_NUM = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

def _metrics_from_string(s: str):
    """
    Parse strings like: "(2.1538, UCM: 44.64% LCM: 17.86% UAS: 62.35% LAS: 47.77%)"
    Returns dict of floats when present.
    """
    out = {}
    mloss = re.search(r"\(\s*(" + _NUM + r")", s)
    if mloss: out["loss"] = float(mloss.group(1))
    for k in ["UCM", "LCM", "UAS", "LAS"]:
        m = re.search(rf"{k}\s*:\s*({_NUM})\s*%?", s, re.I)
        if m:
            out[k] = float(m.group(1))
    return out

def metrics_to_plain(obj):
    """
    Convert SuPar outputs into JSON-serializable dicts/floats/strings.
    Tries to parse UAS/LAS/UCM/LCM from string repr when needed.
    """
    if isinstance(obj, (int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: metrics_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        try:
            s = str(obj)
            parsed = _metrics_from_string(s)
            if parsed:
                return parsed
        except Exception:
            pass
        return [metrics_to_plain(x) for x in obj]
    try:
        s = str(obj)
        parsed = _metrics_from_string(s)
        if parsed:
            return parsed
        return s
    except Exception:
        return repr(obj)

# =========================================================================================
# Step 0: Build parser ONCE (with pretrained XLM-R features), but don't train yet.
#         We still pass EN train/dev so SuPar can build fields/vocabs.
#         Then perform ZERO-SHOT BN evaluation BEFORE training.
# =========================================================================================
print("Building parser (pre-training)...")
tee = Tee(LOG_TXT, mode="w")
with redirect_stdout(tee), redirect_stderr(tee):
    parser = CRF2oDependencyParser.build(
        path=str(MODEL_PT),         # where best checkpoint will later be saved
        train=str(EN_TRAIN),        # used to build fields/vocabs
        dev=str(EN_DEV),
        encoder="bert",
        bert=XLMR,
        finetune=True,
        # NOTE: n_layers=1 uses only the last XLM-R layer (memory friendly).
        # Remove this arg to enable scalar-mix over layers for a small accuracy bump (uses more VRAM).
        n_layers=1,
        seed=42,
        device=DEVICE,
    )

    # Try to enable HF gradient checkpointing to save memory
    try:
        bert_model = parser.model.encoder.bert
        if hasattr(bert_model, "gradient_checkpointing_enable"):
            bert_model.gradient_checkpointing_enable()
        if hasattr(bert_model.config, "gradient_checkpointing"):
            bert_model.config.gradient_checkpointing = True
    except Exception:
        pass

# ---- BN zero-shot (pre-training) ----
metrics = {}
try:
    print("🧪 Zero-shot BN test (pre-training)...")
    with redirect_stdout(tee), redirect_stderr(tee):
        parser.predict(str(BN_TEST), pred=str(BN_PRED_PRE), batch_size=1000, device=DEVICE)
        bn_scores_pre = parser.evaluate(str(BN_TEST), batch_size=1000, device=DEVICE)
    metrics["bn_test_pretrain_raw"] = str(bn_scores_pre)
    metrics["bn_test_pretrain"] = metrics_to_plain(bn_scores_pre)
    print("BN zero-shot predictions:", BN_PRED_PRE)
    print("BN BRU zero-shot test:", bn_scores_pre)
except Exception as e:
    print("⚠️ BN zero-shot prediction/evaluation failed:", e)

tee.flush()
tee.close()

# =========================================================================================
# Step 1: REBUILD parser FRESH for EN training (avoid stale args like 'data' from evaluate)
# =========================================================================================
print("Rebuilding parser (fresh) for EN training...")
tee = Tee(LOG_TXT, mode="a")  # append to the same log
with redirect_stdout(tee), redirect_stderr(tee):
    parser = CRF2oDependencyParser.build(
        path=str(MODEL_PT),
        train=str(EN_TRAIN),
        dev=str(EN_DEV),
        encoder="bert",
        bert=XLMR,
        finetune=True,
        n_layers=1,
        seed=42,
        device=DEVICE,
    )
    # Best-effort: enable HF gradient checkpointing
    try:
        bert_model = parser.model.encoder.bert
        if hasattr(bert_model, "gradient_checkpointing_enable"):
            bert_model.gradient_checkpointing_enable()
        if hasattr(bert_model.config, "gradient_checkpointing"):
            bert_model.config.gradient_checkpointing = True
    except Exception:
        pass

    # Extra guard (if a future SuPar version carries stale keys)
    try:
        if hasattr(parser, "args") and isinstance(parser.args, dict):
            parser.args.pop("data", None)
    except Exception:
        pass

    print("Starting EN (EWT) training...")
    parser.train(
        train=str(EN_TRAIN),
        dev=str(EN_DEV),
        test=str(EN_DEV),   # SuPar expects a real path
        optimizer="adam",
        lr=5e-5,            # encoder LR
        lr_rate=20.0,       # non-encoder head LR multiplier => 1e-3
        warmup=0.1,
        clip=5.0,
        checkpoint=False,
        batch_size=1000,    # micro-batch size in tokens
        update_steps=6,     # gradient accumulation
        buckets=16,
        epochs=25,
        patience=5,
    )

tee.flush()
tee.close()
print(f"✅ Training finished. Model saved at: {MODEL_PT} (exists={MODEL_PT.exists()})")
print(f"Raw log saved to: {LOG_TXT}")

# ---- (Re)load the best checkpoint safely and move to device ----
try:
    if MODEL_PT.exists():
        parser = CRF2oDependencyParser.load(str(MODEL_PT))
        try:
            parser.model.to(DEVICE)
        except Exception:
            pass
except Exception as _e:
    print("⚠️ Reloading best checkpoint failed; continuing with in-memory model:", _e)

# =========================================================================================
# Step 2: Evaluate EN dev (post-training)
# =========================================================================================
try:
    dev_scores = parser.evaluate(str(EN_DEV), batch_size=1000, device=DEVICE)
    metrics["en_dev_raw"] = str(dev_scores)
    metrics["en_dev"] = metrics_to_plain(dev_scores)
    print("EN dev (post-training):", dev_scores)
except Exception as e:
    print("⚠️ Dev evaluation failed:", e)

# =========================================================================================
# Step 3: BN re-test (post-training)
# =========================================================================================
try:
    parser.predict(str(BN_TEST), pred=str(BN_PRED_POST), batch_size=1000, device=DEVICE)
    bn_scores_post = parser.evaluate(str(BN_TEST), batch_size=1000, device=DEVICE)
    metrics["bn_test_trained_raw"] = str(bn_scores_post)
    metrics["bn_test_trained"] = metrics_to_plain(bn_scores_post)
    print("BN predictions after EN training:", BN_PRED_POST)
    print("BN BRU post-training test:", bn_scores_post)
except Exception as e:
    print("⚠️ BN post-training prediction/evaluation failed:", e)

# =========================================================================================
# Step 4: Parse learning curve from train log & save CSV
# =========================================================================================
raw_log = Path(LOG_TXT).read_text(encoding="utf-8")
raw_log = raw_log.replace("\r", "\n")
raw_log = re.sub(r"\x1b\[[0-9;]*m", "", raw_log)  # strip ANSI

epoch, tr_loss, dv_uas, dv_las = [], [], [], []
patterns = [
    # "Epoch 1 ... loss: 1.23 ... UAS: 89.10 ... LAS: 87.55"
    re.compile(
        r"Epoch\s*(\d+)[^\n]*?loss[^0-9-]*(" + _NUM + r")[^\n]*?UAS[^0-9-]*(" + _NUM + r")[^\n]*?LAS[^0-9-]*(" + _NUM + r")",
        re.I,
    ),
    # equals style
    re.compile(
        r"Epoch\s*(\d+)[^\n]*?loss\s*=\s*(" + _NUM + r")[^\n]*?UAS\s*=\s*(" + _NUM + r")[^\n]*?LAS\s*=\s*(" + _NUM + r")",
        re.I,
    ),
]
for line in raw_log.splitlines():
    for pat in patterns:
        m = pat.search(line)
        if m:
            epoch.append(int(m.group(1)))
            tr_loss.append(float(m.group(2)))
            dv_uas.append(float(m.group(3)))
            dv_las.append(float(m.group(4)))
            break

curve_df = pd.DataFrame({
    "epoch": epoch,
    "train_loss": tr_loss,
    "dev_UAS": dv_uas,
    "dev_LAS": dv_las
})
curve_df.to_csv(CURVE_CSV, index=False)
print(f"Learning curve points: {len(curve_df)}  -> {CURVE_CSV}")

# =========================================================================================
# Step 5: Save metrics (JSON-safe)
# =========================================================================================
Path(METRICS_JSON).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved metrics:", METRICS_JSON)

# =========================================================================================
# Step 6: Plots
#   - Training loss per epoch
#   - Dev UAS/LAS per epoch
#   - Correlation matrix among (train_loss, dev_UAS, dev_LAS)
#   - BN pre vs post (UAS/LAS) comparison
# =========================================================================================
if len(curve_df) > 0:
    # 1) Train loss
    plt.figure()
    plt.plot(curve_df["epoch"], curve_df["train_loss"], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Train loss"); plt.title("Training Loss per Epoch")
    plt.grid(True); plt.tight_layout()
    plt.show()

    # 2) Dev UAS & LAS
    plt.figure()
    plt.plot(curve_df["epoch"], curve_df["dev_UAS"], marker="o", label="UAS")
    plt.plot(curve_df["epoch"], curve_df["dev_LAS"], marker="o", label="LAS")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Dev UAS/LAS per Epoch")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # 3) Correlation matrix among metrics
    corr = curve_df[["train_loss", "dev_UAS", "dev_LAS"]].corr(method="pearson")
    print("\nCorrelation matrix (Pearson):\n", corr)

    plt.figure()
    plt.imshow(corr.values, interpolation="nearest")
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=45, ha="right")
    plt.yticks(range(corr.shape[0]), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center")
    plt.title("Correlation Matrix (train_loss, dev_UAS, dev_LAS)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No epoch lines were parsed from the log. SuPar’s logging format may differ across versions.")

# 4) BN pre vs post (UAS/LAS) comparison
bn_pre = metrics.get("bn_test_pretrain", {})
bn_post = metrics.get("bn_test_trained", {})

if "UAS" in bn_pre and "UAS" in bn_post:
    plt.figure()
    plt.bar(["BN zero-shot", "BN post-train"], [bn_pre["UAS"], bn_post["UAS"]])
    plt.ylabel("UAS"); plt.title("BN UAS: Zero-shot vs Post-EN-Training")
    plt.tight_layout()
    plt.show()

if "LAS" in bn_pre and "LAS" in bn_post:
    plt.figure()
    plt.bar(["BN zero-shot", "BN post-train"], [bn_pre["LAS"], bn_post["LAS"]])
    plt.ylabel("LAS"); plt.title("BN LAS: Zero-shot vs Post-EN-Training")
    plt.tight_layout()
    plt.show()

# Print quick deltas to console
if "UAS" in bn_pre and "UAS" in bn_post:
    print(f"ΔBN UAS (post - pre): {bn_post['UAS'] - bn_pre['UAS']:.2f}")
if "LAS" in bn_pre and "LAS" in bn_post:
    print(f"ΔBN LAS (post - pre): {bn_post['LAS'] - bn_pre['LAS']:.2f}")

# ==== Cell 5 ====
outputs_dep/supar_crf2o_xlmr_en/parser.pt

# ==== Cell 6 ====
import torch
from supar.parsers import CRF2oDependencyParser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load your best trained parser
parser = CRF2oDependencyParser.load("outputs_dep/supar_crf2o_xlmr_en/parser.pt")
parser.model.to(DEVICE)

# Your Bengali sentence
sent = "বাংলা একটি ইন্দো-আর্য ভাষা যা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গসহ বাংলাভাষী অঞ্চলে প্রচলিত। এটি বিশ্বের সপ্তম বৃহত্তম ভাষা এবং বাংলাভাষী জাতির প্রধান ভাষা।"

# Run parsing (batch of one sentence)
pred = parser.predict([sent], prob=True, verbose=True, lang='bn')

# Inspect tokens + heads + relations
for tree in pred.sentences:
    print("\n--- Parsed sentence ---")
    for tok in tree:
        print(f"{tok.id}\t{tok.form}\thead={tok.head}\tdeprel={tok.deprel}")

# ==== Cell 7 ====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import torch
from supar.parsers import CRF2oDependencyParser

MODEL_PT = "outputs_dep/supar_crf2o_xlmr_en/parser.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

text = (
    "বাংলা একটি ইন্দো-আর্য ভাষা যা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গসহ বাংলাভাষী অঞ্চলে প্রচলিত। "
    "এটি বিশ্বের সপ্তম বৃহত্তম ভাষা এবং বাংলাভাষী জাতির প্রধান ভাষা।"
)

# --- simple sentence splitter for Bengali (split on danda/question/exclamation) ---
def split_sentences(s: str):
    # Keep the delimiter as its own token (we'll reattach to sentence)
    parts = re.split(r'([।!?])', s.strip())
    sents = []
    cur = ""
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if seg:
            sents.append((seg + (punct or "")).strip())
    return sents

# --- tokenizer: words + keep punctuation as separate tokens ---
# \w in Unicode covers Bengali letters/digits/marks; we also capture hyphenated tokens.
WORD_OR_PUNCT = re.compile(r"[^\W_][\w\u200c\u200d-]*|[^\s\w]", re.UNICODE)

def tokenize(sent: str):
    return WORD_OR_PUNCT.findall(sent)

# Pre-tokenize
sents_tok = [tokenize(s) for s in split_sentences(text)]

print("Pre-tokenized sentences:")
for i, toks in enumerate(sents_tok, 1):
    print(f"{i:>2}: {' '.join(toks)}")

# Load model
parser = CRF2oDependencyParser.load(MODEL_PT)
try:
    parser.model.to(DEVICE)
except Exception:
    pass

# Run prediction with pre-tokenized input: pass list[list[str]] and lang=None
pred = parser.predict(sents_tok, prob=False, verbose=True, lang=None)

# Pretty print tokens, heads, labels
for si, tree in enumerate(pred.sentences, 1):
    print(f"\n--- Parsed sentence {si} ---")
    print("{:>3}  {:<20} {:>4}  {:<10}".format("ID", "FORM", "HEAD", "DEPREL"))
    print("-"*45)
    for tok in tree:
        print("{:>3}  {:<20} {:>4}  {:<10}".format(tok.id, tok.form, tok.head, tok.deprel))

# (Optional) save as CoNLL-U for manual checking/visualization
SAVE_CONLLU = False
if SAVE_CONLLU:
    out_path = "one_para_bn.pred.conllu"
    parser.predict(sents_tok, pred=out_path, lang=None)
    print("\nSaved CoNLL-U to:", out_path)

# ==== Cell 8 ====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, tempfile
from pathlib import Path
import torch
from supar.parsers import CRF2oDependencyParser

MODEL_PT = "outputs_dep/supar_crf2o_xlmr_en/parser.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

text = (
    "বাংলা একটি ইন্দো-আর্য ভাষা যা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গসহ বাংলাভাষী অঞ্চলে প্রচলিত। "
    "এটি বিশ্বের সপ্তম বৃহত্তম ভাষা এবং বাংলাভাষী জাতির প্রধান ভাষা।"
)

# -------- Bengali-friendly sentence split (keep ।?!) --------
def split_sentences(s: str):
    parts = re.split(r'([।!?])', s.strip())
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if seg:
            out.append((seg + (punct or "")).strip())
    return out

# -------- Bengali word tokenizer (no external deps) ----------
# Treat sequences in U+0980–U+09FF as words (includes Bengali letters & marks),
# allow internal hyphens; separate everything else as single-char tokens.
BN_BLOCK = r"\u0980-\u09FF"
WORD_RE = re.compile(fr"[{BN_BLOCK}]+(?:-[{BN_BLOCK}]+)*|\d+|[^\s]", re.UNICODE)

def tokenize_bn(sent: str):
    return WORD_RE.findall(sent)

def to_conllu(sent_tokens, sent_id=1):
    lines = ["# sent_id = {}".format(sent_id)]
    lines.append("# text = " + " ".join(sent_tokens))
    for i, tok in enumerate(sent_tokens, 1):
        # ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
        lines.append(f"{i}\t{tok}\t_\t_\t_\t_\t0\tdep\t_\t_")
    lines.append("")  # sentence boundary
    return "\n".join(lines)

# 1) Pre-tokenize properly
sents = [tokenize_bn(s) for s in split_sentences(text)]
print("Pre-tokenized sentences:")
for i, toks in enumerate(sents, 1):
    print(f"{i:>2}: {' '.join(toks)}")

# 2) Write a minimal temporary CoNLL-U
tmp_dir = tempfile.mkdtemp(prefix="bn_manual_")
in_path = Path(tmp_dir) / "in.conllu"
out_path = Path(tmp_dir) / "pred.conllu"

with open(in_path, "w", encoding="utf-8") as f:
    for sid, toks in enumerate(sents, 1):
        f.write(to_conllu(toks, sent_id=sid))

# 3) Load model and predict (NO stanza, NO lang)
parser = CRF2oDependencyParser.load(MODEL_PT)
try:
    parser.model.to(DEVICE)
except Exception:
    pass

# Important: give it a file path & lang=None
parser.predict(str(in_path), pred=str(out_path), lang=None, verbose=True)

# 4) Read predictions and pretty-print
print(f"\nPredicted CoNLL-U: {out_path}")
cur_sent = 0
print_row = lambda i, form, head, rel: print(f"{i:>3}  {form:<20} {head:>4}  {rel:<12}")
with open(out_path, "r", encoding="utf-8") as f:
    print("\n--- Parsed output ---")
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith("# sent_id"):
            cur_sent += 1
            print(f"\nSentence {cur_sent}")
            print(" ID  FORM                 HEAD  DEPREL")
            print("---- -------------------- ----  ------------")
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10:  # skip odd lines
            continue
        tid, form, _, _, _, _, head, rel, _, _ = cols
        # skip multi-word tokens / empty nodes if ever present
        if "-" in tid or "." in tid:
            continue
        print_row(int(tid), form, head, rel)

print("\nDone. You can open the .conllu to inspect full details.")

# ==== Cell 9 ====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, tempfile
from pathlib import Path
import torch
from supar.parsers import CRF2oDependencyParser

MODEL_PT = "outputs_dep/supar_crf2o_xlmr_en/parser.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

text = (
    "বাংলা একটি ইন্দো-আর্য ভাষা যা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গসহ বাংলাভাষী অঞ্চলে প্রচলিত। "
    "এটি বিশ্বের সপ্তম বৃহত্তম ভাষা এবং বাংলাভাষী জাতির প্রধান ভাষা।"
)

# --- sentence split ---
def split_sentences(s: str):
    parts = re.split(r'([।!?])', s.strip())
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if seg:
            out.append((seg + (punct or "")).strip())
    return out

# --- Bengali tokenizer (keep Bengali word blocks together, punctuation separate) ---
BN_BLOCK = r"\u0980-\u09FF"
WORD_RE = re.compile(fr"[{BN_BLOCK}]+(?:-[{BN_BLOCK}]+)*|\d+|[^\s]", re.UNICODE)
def tokenize_bn(sent: str): return WORD_RE.findall(sent)

def to_conllu(sent_tokens, sent_id=1):
    lines = [f"# sent_id = {sent_id}", "# text = " + " ".join(sent_tokens)]
    for i, tok in enumerate(sent_tokens, 1):
        lines.append(f"{i}\t{tok}\t_\t_\t_\t_\t0\tdep\t_\t_")
    lines.append("")
    return "\n".join(lines)

# 1) pre-tokenize
sents = [tokenize_bn(s) for s in split_sentences(text)]
print("Pre-tokenized sentences:")
for i, toks in enumerate(sents, 1):
    print(f"{i:>2}: {' '.join(toks)}")

# 2) temp CoNLL-U paths
tmp = Path(tempfile.mkdtemp(prefix="bn_manual_"))
in_path  = tmp / "in.conllu"
out_path = tmp / "pred.conllu"
with open(in_path, "w", encoding="utf-8") as f:
    for sid, toks in enumerate(sents, 1):
        f.write(to_conllu(toks, sid))

# 3) load parser
parser = CRF2oDependencyParser.load(MODEL_PT)
try: parser.model.to(DEVICE)
except Exception: pass

# (Patch) ensure transform doesn’t request 'probs'
try:
    tr = parser.transform
    # fields and tgt are lists of field names
    if hasattr(tr, "fields") and isinstance(tr.fields, list):
        tr.fields = [x for x in tr.fields if x != "probs"]
    if hasattr(tr, "tgt") and isinstance(tr.tgt, list):
        tr.tgt = [x for x in tr.tgt if x != "probs"]
except Exception:
    pass

# 4) predict with explicit flags (important!)
parser.predict(str(in_path), pred=str(out_path),
               lang=None, prob=False, mbr=False, tree=False, verbose=True)

# 5) pretty print
print(f"\nPredicted CoNLL-U: {out_path}")
print("\n--- Parsed output ---")
cur = 0
with open(out_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line: continue
        if line.startswith("# sent_id"):
            cur += 1
            print(f"\nSentence {cur}")
            print(" ID  FORM                 HEAD  DEPREL")
            print("---- -------------------- ----  ------------")
            continue
        if line.startswith("#"): continue
        cols = line.split("\t")
        if len(cols) != 10: continue
        tid, form, _, _, _, _, head, rel, _, _ = cols
        if "-" in tid or "." in tid:  # skip MWT/empty nodes
            continue
        print(f"{int(tid):>3}  {form:<20} {head:>4}  {rel:<12}")

print("\nDone. (Open the .conllu file to inspect full details.)")

# ==== Cell 10 ====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, tempfile
from pathlib import Path
import torch
from supar.parsers import CRF2oDependencyParser

MODEL_PT = "outputs_dep/supar_crf2o_xlmr_en/parser.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

text = (
    "বাংলা একটি ইন্দো-আর্য ভাষা যা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গসহ বাংলাভাষী অঞ্চলে প্রচলিত। "
    "এটি বিশ্বের সপ্তম বৃহত্তম ভাষা এবং বাংলাভাষী জাতির প্রধান ভাষা।"
)

# --- sentence split ---
def split_sentences(s: str):
    parts = re.split(r'([।!?])', s.strip())
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if seg:
            out.append((seg + (punct or "")).strip())
    return out

# --- Bengali tokenizer ---
BN_BLOCK = r"\u0980-\u09FF"
WORD_RE = re.compile(fr"[{BN_BLOCK}]+(?:-[{BN_BLOCK}]+)*|\d+|[^\s]", re.UNICODE)
def tokenize_bn(sent: str): return WORD_RE.findall(sent)

def to_conllu(sent_tokens, sent_id=1):
    lines = [f"# sent_id = {sent_id}", "# text = " + " ".join(sent_tokens)]
    for i, tok in enumerate(sent_tokens, 1):
        # ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
        lines.append(f"{i}\t{tok}\t_\t_\t_\t_\t0\tdep\t_\t_")
    lines.append("")  # <- blank line to end the sentence
    return "\n".join(lines)

# 1) pre-tokenize
sents = [tokenize_bn(s) for s in split_sentences(text)]
print("Pre-tokenized sentences:")
for i, toks in enumerate(sents, 1):
    print(f"{i:>2}: {' '.join(toks)}")

# 2) write temp CoNLL-U (with final newline)
tmp = Path(tempfile.mkdtemp(prefix="bn_manual_"))
in_path  = tmp / "in.conllu"
out_path = tmp / "pred.conllu"
with open(in_path, "w", encoding="utf-8") as f:
    for sid, toks in enumerate(sents, 1):
        f.write(to_conllu(toks, sid) + "\n")   # ensure extra newline at EOF

# 3) load model
parser = CRF2oDependencyParser.load(MODEL_PT)
try: parser.model.to(DEVICE)
except Exception: pass

# 4) predict — force single bucket to skip k-means
parser.predict(
    str(in_path),
    pred=str(out_path),
    lang=None,
    prob=False, mbr=False, tree=False,
    buckets=1,
    batch_size=5000,
    verbose=True
)

# 5) pretty print
print(f"\nPredicted CoNLL-U: {out_path}")
print("\n--- Parsed output ---")
cur = 0
with open(out_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line: continue
        if line.startswith("# sent_id"):
            cur += 1
            print(f"\nSentence {cur}")
            print(" ID  FORM                 HEAD  DEPREL")
            print("---- -------------------- ----  ------------")
            continue
        if line.startswith("#"): continue
        cols = line.split("\t")
        if len(cols) != 10: continue
        tid, form, _, _, _, _, head, rel, _, _ = cols
        if "-" in tid or "." in tid:  # skip MWT/empty nodes
            continue
        print(f"{int(tid):>3}  {form:<20} {head:>4}  {rel:<12}")

print("\nDone.")

# ==== Cell 11 ====
# after: parser = CRF2oDependencyParser.load(MODEL_PT)

def _strip_field(transform, name: str):
    """Remove a field name from SuPar's transform bookkeeping (list/tuple)."""
    for attr in ("fields", "src", "tgt"):
        if hasattr(transform, attr):
            v = getattr(transform, attr)
            if isinstance(v, (list, tuple)):
                nv = [x for x in v if x != name]
                setattr(transform, attr, type(v)(nv))
    # Some versions cache a flattened copy; wipe it so it's recomputed
    if hasattr(transform, "_flattened_fields"):
        try:
            setattr(transform, "_flattened_fields", None)
        except Exception:
            pass

# --- apply the patch ---
tr = parser.transform
# (optional) see before/after
try:
    print("BEFORE fields:", getattr(tr, "fields", None))
    print("BEFORE src:",    getattr(tr, "src", None))
    print("BEFORE tgt:",    getattr(tr, "tgt", None))
except Exception:
    pass

_strip_field(tr, "probs")   # <-- the culprit in your traceback

try:
    print("AFTER  fields:", getattr(tr, "fields", None))
    print("AFTER  src:",    getattr(tr, "src", None))
    print("AFTER  tgt:",    getattr(tr, "tgt", None))
except Exception:
    pass

# ==== Cell 12 ====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual one-paragraph parse with SuPar (CRF2oDependencyParser) for Bengali.
- No Stanza needed (we pre-tokenize)
- Strips lingering 'probs' field from the saved transform
- Forces single-bucket inference to avoid k-means issues
"""
import re
import tempfile
from pathlib import Path
import torch
from supar.parsers import CRF2oDependencyParser

# ==== CONFIG ====
MODEL_PT = "outputs_dep/supar_crf2o_xlmr_en/parser.pt"
TEXT = (
    "বাংলা একটি ইন্দো-আর্য ভাষা যা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গসহ বাংলাভাষী অঞ্চলে প্রচলিত। "
    "এটি বিশ্বের সপ্তম বৃহত্তম ভাষা এবং বাংলাভাষী জাতির প্রধান ভাষা।"
)
FORCE_SAVE_CONLLU = True  # also write a local copy: ./bn_manual.pred.conllu

# ==== UTILITIES ====
def split_sentences(s: str):
    """Split on Bengali danda/question/exclamation, keep punctuation attached."""
    parts = re.split(r'([।!?])', s.strip())
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if seg:
            out.append((seg + (punct or "")).strip())
    return out

# Bengali word block (U+0980..U+09FF). Keep hyphenated Bengali words, digits, or single non-space chars (punct).
BN_BLOCK = r"\u0980-\u09FF"
WORD_RE = re.compile(fr"[{BN_BLOCK}]+(?:-[{BN_BLOCK}]+)*|\d+|[^\s]", re.UNICODE)

def tokenize_bn(sent: str):
    return WORD_RE.findall(sent)

def to_conllu(sent_tokens, sent_id=1):
    lines = [f"# sent_id = {sent_id}", "# text = " + " ".join(sent_tokens)]
    for i, tok in enumerate(sent_tokens, 1):
        # ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
        lines.append(f"{i}\t{tok}\t_\t_\t_\t_\t0\tdep\t_\t_")
    lines.append("")  # blank line to end the sentence
    return "\n".join(lines)

def strip_field_everywhere(transform, name: str):
    """Robustly remove a field name from SuPar's transform bookkeeping."""
    for attr in ("fields", "src", "tgt"):
        if hasattr(transform, attr):
            v = getattr(transform, attr)
            if isinstance(v, (list, tuple)):
                nv = [x for x in list(v) if x != name]
                try:
                    setattr(transform, attr, type(v)(nv))
                except Exception:
                    setattr(transform, attr, nv)
    # clear any cached flattened lists
    for cache_attr in ("_flattened_fields", "flattened", "__flattened"):
        if hasattr(transform, cache_attr):
            try:
                setattr(transform, cache_attr, None)
            except Exception:
                pass
    # delete a stray attribute if present
    if hasattr(transform, name):
        try:
            delattr(transform, name)
        except Exception:
            pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Pre-tokenize
    sents = [tokenize_bn(s) for s in split_sentences(TEXT)]
    print("Pre-tokenized sentences:")
    for i, toks in enumerate(sents, 1):
        print(f"{i:>2}: {' '.join(toks)}")

    # 2) Write temp CoNLL-U (ensure final newline)
    tmp = Path(tempfile.mkdtemp(prefix="bn_manual_"))
    in_path  = tmp / "in.conllu"
    out_path = tmp / "pred.conllu"
    with open(in_path, "w", encoding="utf-8") as f:
        for sid, toks in enumerate(sents, 1):
            f.write(to_conllu(toks, sid) + "\n")

    # 3) Load parser & move to device
    parser = CRF2oDependencyParser.load(MODEL_PT)
    try:
        parser.model.to(device)
    except Exception:
        pass

    # 4) Patch transform: remove 'probs' everywhere (prevents AttributeError)
    tr = parser.transform
    # debug (optional): uncomment next lines to inspect pre/after
    # print("[DEBUG] BEFORE fields:", getattr(tr, "fields", None))
    # print("[DEBUG] BEFORE src   :", getattr(tr, "src", None))
    # print("[DEBUG] BEFORE tgt   :", getattr(tr, "tgt", None))

    strip_field_everywhere(tr, "probs")

    # print("[DEBUG] AFTER  fields:", getattr(tr, "fields", None))
    # print("[DEBUG] AFTER  src   :", getattr(tr, "src", None))
    # print("[DEBUG] AFTER  tgt   :", getattr(tr, "tgt", None))

    # 5) Predict — skip k-means, skip probability/extra decoding
    parser.predict(
        str(in_path),
        pred=str(out_path),
        lang=None,
        prob=False, mbr=False, tree=False,
        buckets=1, batch_size=5000,
        verbose=True
    )

    # 6) Pretty-print predictions
    print(f"\nPredicted CoNLL-U: {out_path}")
    print("\n--- Parsed output ---")
    cur = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# sent_id"):
                cur += 1
                print(f"\nSentence {cur}")
                print(" ID  FORM                 HEAD  DEPREL")
                print("---- -------------------- ----  ------------")
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tid, form, _, _, _, _, head, rel, _, _ = cols
            if "-" in tid or "." in tid:
                continue
            print(f"{int(tid):>3}  {form:<20} {head:>4}  {rel:<12}")

    # 7) Optionally save a convenient copy next to the script
    if FORCE_SAVE_CONLLU:
        local_out = Path("bn_manual.pred.conllu")
        local_out.write_text(Path(out_path).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nSaved copy to: {local_out.resolve()}")

if __name__ == "__main__":
    main()

# ==== Cell 13 ====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual parse of a Bengali paragraph with a saved SuPar CRF2o model.
- Pre-tokenizes (no Stanza).
- Removes stray 'probs' from transform.fields (does NOT touch read-only src/tgt).
- Forces single-bucket inference to avoid k-means issues.
"""
import re
import tempfile
from pathlib import Path
import torch
from supar.parsers import CRF2oDependencyParser

# === CONFIG ===
MODEL_PT = "outputs_dep/supar_crf2o_xlmr_en/parser.pt"
TEXT = (
    "বাংলা একটি ইন্দো-আর্য ভাষা যা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গসহ বাংলাভাষী অঞ্চলে প্রচলিত। "
    "এটি বিশ্বের সপ্তম বৃহত্তম ভাষা এবং বাংলাভাষী জাতির প্রধান ভাষা।"
)
SAVE_COPY = True  # write bn_manual.pred.conllu next to the script

# --- simple Bengali sentence split (keep ।?! as sentence-final) ---
def split_sentences(s: str):
    parts = re.split(r'([।!?])', s.strip())
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if seg:
            out.append((seg + (punct or "")).strip())
    return out

# --- Bengali tokenizer: keep U+0980–U+09FF word blocks (allow hyphens), digits, or single punctuation ---
BN_BLOCK = r"\u0980-\u09FF"
WORD_RE = re.compile(fr"[{BN_BLOCK}]+(?:-[{BN_BLOCK}]+)*|\d+|[^\s]", re.UNICODE)
def tokenize_bn(sent: str): return WORD_RE.findall(sent)

def to_conllu(tokens, sent_id=1):
    lines = [f"# sent_id = {sent_id}", "# text = " + " ".join(tokens)]
    for i, tok in enumerate(tokens, 1):
        # ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
        lines.append(f"{i}\t{tok}\t_\t_\t_\t_\t0\tdep\t_\t_")
    lines.append("")  # blank line terminator
    return "\n".join(lines)

def strip_probs_from_fields(transform):
    """
    Only remove the *name* 'probs' from transform.fields.
    DO NOT touch read-only properties like src/tgt.
    """
    try:
        fields = getattr(transform, "fields", None)
        if fields is None:
            return
        # normalize to list, filter, then write back using safest method
        flist = list(fields)
        if "probs" in flist:
            flist = [x for x in flist if x != "probs"]
            try:
                transform.fields = type(fields)(flist)  # preserve list/tuple type
            except Exception:
                # fallback: write directly into __dict__
                transform.__dict__["fields"] = flist
        # clear any cached flattened fields so SuPar recomputes them
        for cache_attr in ("_flattened_fields", "flattened", "__flattened"):
            if hasattr(transform, cache_attr):
                try:
                    setattr(transform, cache_attr, None)
                except Exception:
                    try:
                        delattr(transform, cache_attr)
                    except Exception:
                        pass
    except Exception:
        # Last resort: if transform has attribute named 'probs', drop it so getattr won't fail
        if hasattr(transform, "probs"):
            try:
                delattr(transform, "probs")
            except Exception:
                pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Pre-tokenize the paragraph
    sents = [tokenize_bn(s) for s in split_sentences(TEXT)]
    print("Pre-tokenized sentences:")
    for i, toks in enumerate(sents, 1):
        print(f"{i:>2}: {' '.join(toks)}")

    # 2) Write a clean temporary CoNLL-U file
    tmpdir = Path(tempfile.mkdtemp(prefix="bn_manual_"))
    in_path  = tmpdir / "in.conllu"
    out_path = tmpdir / "pred.conllu"
    with open(in_path, "w", encoding="utf-8") as f:
        for sid, toks in enumerate(sents, 1):
            f.write(to_conllu(toks, sid) + "\n")  # ensure final newline

    # 3) Load model and move to device
    parser = CRF2oDependencyParser.load(MODEL_PT)
    try:
        parser.model.to(device)
    except Exception:
        pass

    # 4) Remove stray 'probs' ONLY from fields (do not touch src/tgt)
    strip_probs_from_fields(parser.transform)

    # 5) Predict: skip probabilities and bucketing
    parser.predict(
        str(in_path),
        pred=str(out_path),
        lang=None,
        prob=False, mbr=False, tree=False,
        buckets=1, batch_size=5000,
        verbose=True
    )

    # 6) Pretty-print a compact view
    print(f"\nPredicted CoNLL-U: {out_path}")
    print("\n--- Parsed output ---")
    cur = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# sent_id"):
                cur += 1
                print(f"\nSentence {cur}")
                print(" ID  FORM                 HEAD  DEPREL")
                print("---- -------------------- ----  ------------")
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tid, form, *_rest, head, rel, _deps, _misc = cols[0], cols[1], cols[2:7], cols[6], cols[7], cols[8], cols[9]
            if "-" in tid or "." in tid:  # skip MWT/empty nodes if ever present
                continue
            print(f"{int(tid):>3}  {form:<20} {head:>4}  {rel:<12}")

    if SAVE_COPY:
        local = Path("bn_manual.pred.conllu")
        local.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nSaved copy: {local.resolve()}")

if __name__ == "__main__":
    main()

# ==== Cell 14 ====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual parse of a Bengali paragraph with a saved SuPar CRF2o model.
- Pre-tokenizes (no Stanza).
- Removes stray 'probs' from transform.fields (does NOT touch read-only src/tgt).
- Forces single-bucket inference and tree decoding to get a proper UD tree.
"""
import re
import tempfile
from pathlib import Path
import torch
from supar.parsers import CRF2oDependencyParser

# === CONFIG ===
MODEL_PT = "outputs_dep/supar_crf2o_xlmr_en/parser.pt"
TEXT = (
    "বাংলা একটি ইন্দো-আর্য ভাষা যা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গসহ বাংলাভাষী অঞ্চলে প্রচলিত। "
    "এটি বিশ্বের সপ্তম বৃহত্তম ভাষা এবং বাংলাভাষী জাতির প্রধান ভাষা।"
)
SAVE_COPY = True  # write bn_manual.pred.conllu next to the script

# --- simple Bengali sentence split (keep ।?! as sentence-final) ---
def split_sentences(s: str):
    parts = re.split(r'([।!?])', s.strip())
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if seg:
            out.append((seg + (punct or "")).strip())
    return out

# --- Bengali tokenizer: keep U+0980–U+09FF word blocks (allow hyphens), digits, or single punctuation ---
BN_BLOCK = r"\u0980-\u09FF"
WORD_RE = re.compile(fr"[{BN_BLOCK}]+(?:-[{BN_BLOCK}]+)*|\d+|[^\s]", re.UNICODE)
def tokenize_bn(sent: str): return WORD_RE.findall(sent)

def to_conllu(tokens, sent_id=1):
    lines = [f"# sent_id = {sent_id}", "# text = " + " ".join(tokens)]
    for i, tok in enumerate(tokens, 1):
        # ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
        lines.append(f"{i}\t{tok}\t_\t_\t_\t_\t0\tdep\t_\t_")
    lines.append("")  # blank line terminator
    return "\n".join(lines)

def strip_probs_from_fields(transform):
    """
    Only remove the *name* 'probs' from transform.fields.
    DO NOT touch read-only properties like src/tgt.
    """
    try:
        fields = getattr(transform, "fields", None)
        if fields is None:
            return
        flist = list(fields)
        if "probs" in flist:
            flist = [x for x in flist if x != "probs"]
            try:
                transform.fields = type(fields)(flist)  # preserve list/tuple
            except Exception:
                # fallback: write into __dict__ if setter is missing
                transform.__dict__["fields"] = flist
        # clear any cached flattened fields so SuPar recomputes them
        for cache_attr in ("_flattened_fields", "flattened", "__flattened"):
            if hasattr(transform, cache_attr):
                try:
                    setattr(transform, cache_attr, None)
                except Exception:
                    try:
                        delattr(transform, cache_attr)
                    except Exception:
                        pass
    except Exception:
        # last resort: if transform has attribute literally named 'probs', drop it
        if hasattr(transform, "probs"):
            try:
                delattr(transform, "probs")
            except Exception:
                pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Pre-tokenize the paragraph
    sents = [tokenize_bn(s) for s in split_sentences(TEXT)]
    print("Pre-tokenized sentences:")
    for i, toks in enumerate(sents, 1):
        print(f"{i:>2}: {' '.join(toks)}")

    # 2) Write a clean temporary CoNLL-U file (ensure final newline)
    tmpdir = Path(tempfile.mkdtemp(prefix="bn_manual_"))
    in_path  = tmpdir / "in.conllu"
    out_path = tmpdir / "pred.conllu"
    with open(in_path, "w", encoding="utf-8") as f:
        for sid, toks in enumerate(sents, 1):
            f.write(to_conllu(toks, sid) + "\n")

    # 3) Load model and move to device
    parser = CRF2oDependencyParser.load(MODEL_PT)
    try:
        parser.model.to(device)
    except Exception:
        pass

    # 4) Remove stray 'probs' ONLY from fields (do not touch src/tgt)
    strip_probs_from_fields(parser.transform)

    # 5) Predict — enforce tree, skip probabilities, avoid k-means bucketing
    parser.predict(
        str(in_path),
        pred=str(out_path),
        lang=None,
        prob=False, mbr=False,
        tree=True,            # <- enforce single-root tree (head=0, deprel=root)
        proj=False,           # allow non-projective (UD-friendly)
        buckets=1, batch_size=5000,
        verbose=True
    )

    # 6) Pretty-print a compact view
    print(f"\nPredicted CoNLL-U: {out_path}")
    print("\n--- Parsed output ---")
    cur = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# sent_id"):
                cur += 1
                print(f"\nSentence {cur}")
                print(" ID  FORM                 HEAD  DEPREL")
                print("---- -------------------- ----  ------------")
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tid, form, lemma, upos, xpos, feats, head, rel, deps, misc = cols
            if "-" in tid or "." in tid:  # skip MWT/empty nodes if ever present
                continue
            print(f"{int(tid):>3}  {form:<20} {head:>4}  {rel:<12}")

    # 7) Sanity check: count roots & self-loops
    roots = self_loops = sents_count = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# sent_id"):
                sents_count += 1
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tid, head, rel = cols[0], cols[6], cols[7]
            if "-" in tid or "." in tid:
                continue
            if head == "0" and rel.lower() == "root":
                roots += 1
            if tid == head:
                self_loops += 1
    print(f"\n[check] roots={roots}/{sents_count}, self_loops={self_loops}  (expect roots={sents_count}, self_loops=0)")

    if SAVE_COPY:
        local = Path("bn_manual.pred.conllu")
        local.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nSaved copy: {local.resolve()}")

if __name__ == "__main__":
    main()

# ==== Cell 15 ====
gold = read_conllu_rows(BN_TEST)
pred = read_conllu_rows(PRED_BN)
assert len(gold)==len(pred), "Mismatch #sents"

labels = sorted(list({x for r in gold for x in r["deprel"]} | {x for r in pred for x in r["deprel"]}))
lab2i = {l:i for i,l in enumerate(labels)}
cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
for g, p in zip(gold, pred):
    n = min(len(g["deprel"]), len(p["deprel"]))
    for i in range(n):
        cm[lab2i[g["deprel"][i]], lab2i[p["deprel"][i]]] += 1

cmn = cm/(cm.sum(axis=1, keepdims=True)+1e-9)
plt.figure(figsize=(10,8))
sns.heatmap(cmn, xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.title("BN-BRU DEPREL Confusion (Zero-shot)"); plt.xlabel("Pred"); plt.ylabel("Gold")
plt.tight_layout(); plt.show()

corr = np.corrcoef(cmn + 1e-12)
plt.figure(figsize=(8,6))
sns.heatmap(corr, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
plt.title("DEPREL Row Correlation"); plt.tight_layout(); plt.show()

# ==== Cell 16 ====
gimport re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

BN_TEST = "data/ud/bn_bru/bn_bru-test.conllu"                 # gold
PRED_BN = "outputs_dep/supar_crf2o_xlmr_en/bn_bru-test.posttrain.pred.conllu"  # pred

def read_conllu_rows(path, strip_subtypes=True):
    """
    Returns a list of sentences; each sentence is a dict with:
      - 'deprel': [list of dependency labels per token]
    Skips multi-word tokens (1-2) and empty nodes (1.1).
    """
    sents = []
    cur_deps = []
    re_id_ok = re.compile(r"^\d+$")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_deps:
                    sents.append({"deprel": cur_deps})
                    cur_deps = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tid = cols[0]
            if not re_id_ok.match(tid):
                # skip MWT and empty nodes
                continue
            rel = cols[7]
            if strip_subtypes and ":" in rel:
                rel = rel.split(":", 1)[0]
            cur_deps.append(rel)
    if cur_deps:
        sents.append({"deprel": cur_deps})
    return sents

# ---- read data ----
gold = read_conllu_rows(BN_TEST, strip_subtypes=True)
pred = read_conllu_rows(PRED_BN, strip_subtypes=True)
assert len(gold) == len(pred), f"Mismatch #sents gold={len(gold)} pred={len(pred)}"

# ---- label set (union), ordered by gold frequency desc then alpha ----
gold_counts = Counter([rel for s in gold for rel in s["deprel"]])
pred_counts = Counter([rel for s in pred for rel in s["deprel"]])
all_labels = set(gold_counts) | set(pred_counts)
labels = sorted(all_labels, key=lambda x: (-gold_counts.get(x, 0), x))
lab2i = {l: i for i, l in enumerate(labels)}

# ---- confusion matrix ----
cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
for g, p in zip(gold, pred):
    n = min(len(g["deprel"]), len(p["deprel"]))
    for i in range(n):
        gi = lab2i[g["deprel"][i]]
        pi = lab2i[p["deprel"][i]]
        cm[gi, pi] += 1

# ---- row-normalized matrix (gold rows sum to 1) ----
row_sums = cm.sum(axis=1, keepdims=True)
cmn = cm / np.clip(row_sums, 1, None)  # avoid division by 0

# ---- plots ----
plt.figure(figsize=(max(10, 0.5*len(labels)), max(8, 0.5*len(labels))))
sns.heatmap(cmn, xticklabels=labels, yticklabels=labels, cmap="Blues", vmin=0, vmax=1)
plt.title("BN-BRU DEPREL Confusion (row-normalized; gold on rows)")
plt.xlabel("Predicted label")
plt.ylabel("Gold label")
plt.tight_layout()
plt.show()

# Row correlation (how similar each gold label's confusion distribution is to others)
# Add tiny epsilon & handle NaNs from constant rows
eps = 1e-12
corr = np.corrcoef(cmn + eps)
corr = np.nan_to_num(corr, nan=0.0)

plt.figure(figsize=(max(8, 0.45*len(labels)), max(6, 0.45*len(labels))))
sns.heatmap(corr, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
plt.title("DEPREL Row Correlation")
plt.tight_layout()
plt.show()

# ---- quick textual summary ----
print("Top-10 gold labels by frequency:")
for lab, cnt in gold_counts.most_common(10):
    acc = (cm[lab2i[lab], lab2i[lab]] / max(1, cm[lab2i[lab], :].sum())) * 100
    print(f"{lab:>10s}  count={cnt:>5d}  diag-acc={acc:5.1f}%")

# ==== Cell 17 ====
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

BN_TEST = "data/ud/bn_bru/bn_bru-test.conllu"                 # gold
PRED_BN = "outputs_dep/supar_crf2o_xlmr_en/bn_bru-test.posttrain.pred.conllu"  # pred

def read_conllu_rows(path, strip_subtypes=True):
    """
    Returns a list of sentences; each sentence is a dict with:
      - 'deprel': [list of dependency labels per token]
    Skips multi-word tokens (1-2) and empty nodes (1.1).
    """
    sents = []
    cur_deps = []
    re_id_ok = re.compile(r"^\d+$")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_deps:
                    sents.append({"deprel": cur_deps})
                    cur_deps = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tid = cols[0]
            if not re_id_ok.match(tid):
                # skip MWT and empty nodes
                continue
            rel = cols[7]
            if strip_subtypes and ":" in rel:
                rel = rel.split(":", 1)[0]
            cur_deps.append(rel)
    if cur_deps:
        sents.append({"deprel": cur_deps})
    return sents

# ---- read data ----
gold = read_conllu_rows(BN_TEST, strip_subtypes=True)
pred = read_conllu_rows(PRED_BN, strip_subtypes=True)
assert len(gold) == len(pred), f"Mismatch #sents gold={len(gold)} pred={len(pred)}"

# ---- label set (union), ordered by gold frequency desc then alpha ----
gold_counts = Counter([rel for s in gold for rel in s["deprel"]])
pred_counts = Counter([rel for s in pred for rel in s["deprel"]])
all_labels = set(gold_counts) | set(pred_counts)
labels = sorted(all_labels, key=lambda x: (-gold_counts.get(x, 0), x))
lab2i = {l: i for i, l in enumerate(labels)}

# ---- confusion matrix ----
cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
for g, p in zip(gold, pred):
    n = min(len(g["deprel"]), len(p["deprel"]))
    for i in range(n):
        gi = lab2i[g["deprel"][i]]
        pi = lab2i[p["deprel"][i]]
        cm[gi, pi] += 1

# ---- row-normalized matrix (gold rows sum to 1) ----
row_sums = cm.sum(axis=1, keepdims=True)
cmn = cm / np.clip(row_sums, 1, None)  # avoid division by 0

# ---- plots ----
plt.figure(figsize=(max(10, 0.5*len(labels)), max(8, 0.5*len(labels))))
sns.heatmap(cmn, xticklabels=labels, yticklabels=labels, cmap="Blues", vmin=0, vmax=1)
plt.title("BN-BRU DEPREL Confusion (row-normalized; gold on rows)")
plt.xlabel("Predicted label")
plt.ylabel("Gold label")
plt.tight_layout()
plt.show()

# Row correlation (how similar each gold label's confusion distribution is to others)
# Add tiny epsilon & handle NaNs from constant rows
eps = 1e-12
corr = np.corrcoef(cmn + eps)
corr = np.nan_to_num(corr, nan=0.0)

plt.figure(figsize=(max(8, 0.45*len(labels)), max(6, 0.45*len(labels))))
sns.heatmap(corr, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
plt.title("DEPREL Row Correlation")
plt.tight_layout()
plt.show()

# ---- quick textual summary ----
print("Top-10 gold labels by frequency:")
for lab, cnt in gold_counts.most_common(10):
    acc = (cm[lab2i[lab], lab2i[lab]] / max(1, cm[lab2i[lab], :].sum())) * 100
    print(f"{lab:>10s}  count={cnt:>5d}  diag-acc={acc:5.1f}%")

