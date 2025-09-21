# EN→BN Sentiment Classification — Cross-Lingual Transfer (XLM-RoBERTa Base)

This branch documents a Kaggle run where we fine-tuned **XLM-RoBERTa base** on **English (EN)** and evaluated cross-lingual transfer on **Bangla (BN)** for **sentence-level sentiment**.

- **Repo:** `asifmr7/Task-Specific-Cross-Lingual-Transferability-into-Bangla-A-Multi-Source-and-Dialect-Aware-Study`
- **Branch:** `cross-lingual-transfer-en-bn-sentiment-classification`
- **Saved best model (in Kaggle):** `/kaggle/working/best_xlmr_en2bn_sentiment`

> Large model files are **not committed**. Use the offline model folder below or publish your model as a Kaggle dataset / HF snapshot.

---

## What’s included here

- `bn_eval_preds.csv` — BN test predictions (text, gold label if available, pred label, probabilities)
- `RUN_SUMMARY.md` — short notes of the run
- `.gitignore` — ignores large/binary artifacts

*(If present in future commits on this branch: `session_export.ipynb`, `session_export.html`, `metrics.json`.)*

---

## Result snapshot (from this run)

BN test (after EN fine-tuning):

- **Accuracy:** `0.8297620055899043`
- **F1 (binary avg):** `0.8824698865629751`
- Precision ≈ `0.8772`, Recall ≈ `0.8878`

> Numbers can vary with seed, splits, preprocessing.

---

## Offline model setup (works without internet in Kaggle)

Prepare once **on your PC**, then upload to Kaggle as a Dataset.

```bash
pip install huggingface_hub transformers

python - <<'PY'
import os, zipfile
from huggingface_hub import snapshot_download

OUT = "xlm-roberta-base_pytorch_model"
snapshot_download(
    repo_id="xlm-roberta-base",
    local_dir=OUT,
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json","pytorch_model.bin","model.safetensors",
        "tokenizer.json","tokenizer_config.json","special_tokens_map.json",
        "spiece.model","sentencepiece.bpe.model","vocab.json","merges.txt","*.md","*.txt"
    ],
)

with zipfile.ZipFile(OUT + ".zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUT):
        for f in files:
            p = os.path.join(root, f)
            zf.write(p, os.path.relpath(p, start=os.path.dirname(OUT)))

print("Saved:", os.path.abspath(OUT), "and", os.path.abspath(OUT + ".zip"))
PY
