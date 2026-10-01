#!/usr/bin/env python3
"""
compute_heldout_wer.py - WER on test samples that don't overlap with training.

Filters to the non-overlapping sample IDs, runs punctuation-stripped jiwer
on all 3 models, and saves results to data/kaggle_benchmark/analysis/heldout_wer.csv.

Usage:
    python scripts/compute_heldout_wer.py
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

import jiwer
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from accent_labs_config import REPO_ROOT


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def main() -> None:
    repo_root = REPO_ROOT

    train = pd.read_csv(repo_root / "combined_training_manifest.csv")
    test = pd.read_csv(repo_root / "data/kaggle_benchmark/dataset/test_samples.csv")

    train_ids = set(train["sample_id"].astype(str))
    test_ids = set(test["sample_id"].astype(str))
    clean_ids = test_ids - train_ids

    clean_test = test[test["sample_id"].astype(str).isin(clean_ids)]
    print(f"Held-out samples: {len(clean_test)}")

    model_out_dir = repo_root / "data/kaggle_benchmark/model_outputs"
    hypothesis_files = sorted(model_out_dir.glob("*_hypotheses.csv"))

    results = []
    for hyp_file in hypothesis_files:
        model_id = hyp_file.stem.replace("_hypotheses", "")
        df_hyp = pd.read_csv(hyp_file)
        df_hyp["sample_id"] = df_hyp["sample_id"].astype(str)
        df_hyp_filtered = df_hyp[df_hyp["sample_id"].isin(clean_ids)]

        total_wer = 0.0
        count = 0
        for _, row in df_hyp_filtered.iterrows():
            sid = row["sample_id"]
            ref_row = clean_test[clean_test["sample_id"].astype(str) == sid]
            if ref_row.empty:
                continue
            ref_text = str(ref_row.iloc[0]["text"])
            hyp_text = str(row.get("hypothesis_text", ""))
            if hyp_text.startswith("[ERROR"):
                continue
            try:
                w = jiwer.wer(_strip_punctuation(ref_text), _strip_punctuation(hyp_text))
                total_wer += w
                count += 1
            except Exception:
                pass

        avg_wer = total_wer / count if count > 0 else 0.0
        results.append({"model_id": model_id, "held_out_wer": round(avg_wer, 4), "n": count})
        print(f"  {model_id}: WER={avg_wer:.4f}  (n={count})")

    out_dir = repo_root / "data/kaggle_benchmark/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "heldout_wer.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()