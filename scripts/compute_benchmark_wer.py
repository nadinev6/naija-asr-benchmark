#!/usr/bin/env python3
"""
compute_benchmark_wer.py - Step 3: WER/CER calculation with word-level alignment.

For each model output, computes:
  - Raw WER (standard jiwer)
  - Word-level alignment (ref_word, hyp_word, wer_op: equal/substitute/delete/insert)
  - Per-word token log probability (from Whisper output)

Output: data/kaggle_benchmark/annotations/word_alignment.csv

Usage:
    python scripts/compute_benchmark_wer.py [--config CONFIG]
    python scripts/compute_benchmark_wer.py --samples CSV --model-outputs DIR --output-dir DIR
    python scripts/compute_benchmark_wer.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import jiwer
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from accent_labs_config import REPO_ROOT, load_config, resolve_path


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def compute_word_alignment(
    reference: str,
    hypothesis: str,
    sample_id: str,
    model_id: str,
    accent_background: str,
    token_logprobs: list[float] | None = None,
) -> list[dict]:
    """Compute word-level alignment between reference and hypothesis.

    Uses jiwer.process_words alignment chunks. Each chunk describes a span
    of reference vs hypothesis words. Returns one row per reference word,
    plus rows for inserted hypothesis words.
    """
    ref_clean = _strip_punctuation(reference.lower().strip())
    hyp_clean = _strip_punctuation(hypothesis.lower().strip())

    if not ref_clean:
        return []

    measures = jiwer.process_words(ref_clean, hyp_clean)

    # measures.references: [[ref_word0, ref_word1, ...]]  (one inner list per segment)
    # measures.hypotheses: [[hyp_word0, hyp_word1, ...]]
    # measures.alignments: [[chunk0, chunk1, ...]]
    ref_words = measures.references[0] if measures.references else []
    hyp_words = measures.hypotheses[0] if measures.hypotheses else []
    chunks = measures.alignments[0] if measures.alignments else []

    # Map: for each reference word index, what operation applies?
    ref_idx_to_op: dict[int, str] = {}
    ref_idx_to_hyp_idx: dict[int, int] = {}

    for chunk in chunks:
        chunk_type = chunk.type  # equal | substitute | delete | insert
        for ri in range(chunk.ref_start_idx, chunk.ref_end_idx):
            ref_idx_to_op[ri] = chunk_type
            if chunk_type in ("equal", "substitute"):
                # One-to-one mapping between ref and hyp in the range
                hi = chunk.hyp_start_idx + (ri - chunk.ref_start_idx)
                ref_idx_to_hyp_idx[ri] = hi

    rows = []
    for ri, rw in enumerate(ref_words):
        op = ref_idx_to_op.get(ri, "delete")
        hi = ref_idx_to_hyp_idx.get(ri, -1)
        hyp_word = hyp_words[hi] if 0 <= hi < len(hyp_words) else ""

        row = {
            "sample_id": sample_id,
            "model_id": model_id,
            "accent_background": accent_background,
            "word_index": ri,
            "ref_word": rw,
            "hyp_word": hyp_word,
            "wer_op": op,
            "token_logprob": token_logprobs[hi] if token_logprobs and 0 <= hi < len(token_logprobs) else None,
        }
        rows.append(row)

    # Inserted words at the end (trailing insertions after last ref word)
    for chunk in chunks:
        if chunk.type == "insert":
            for hi in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if hi < len(hyp_words):
                    rows.append({
                        "sample_id": sample_id,
                        "model_id": model_id,
                        "accent_background": accent_background,
                        "word_index": len(rows),
                        "ref_word": "",
                        "hyp_word": hyp_words[hi],
                        "wer_op": "insert",
                        "token_logprob": token_logprobs[hi] if token_logprobs and hi < len(token_logprobs) else None,
                    })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WER with word-level alignment.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--samples", type=str, default=None,
                        help="Path to test_samples.csv")
    parser.add_argument("--model-outputs", type=str, default=None,
                        help="Directory with model hypothesis CSVs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for word_alignment.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = REPO_ROOT

    samples_path = resolve_path(repo_root, args.samples or "data/kaggle_benchmark/dataset/test_samples.csv")
    if not samples_path.is_file():
        sys.exit(f"[ERROR] Samples CSV not found: {samples_path}")

    model_out_dir = resolve_path(repo_root, args.model_outputs or "data/kaggle_benchmark/model_outputs")
    if not model_out_dir.is_dir():
        sys.exit(f"[ERROR] Model outputs directory not found: {model_out_dir}")

    out_dir = resolve_path(repo_root, args.output_dir or "data/kaggle_benchmark/annotations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference
    df_samples = pd.read_csv(samples_path)
    reference_map = dict(zip(df_samples["sample_id"], df_samples["text"]))
    accent_map = dict(zip(df_samples["sample_id"], df_samples["accent_background"]))
    print(f"Loaded {len(df_samples)} reference samples")

    # Find model output files
    hypothesis_files = sorted(model_out_dir.glob("*_hypotheses.csv"))
    if not hypothesis_files:
        sys.exit(f"[ERROR] No hypothesis CSV files found in {model_out_dir}")

    print(f"Found {len(hypothesis_files)} model output files:")
    for f in hypothesis_files:
        print(f"  {f.name}")

    if args.dry_run:
        print(f"\n[DRY-RUN] Would process and output to {out_dir / 'word_alignment.csv'}")
        return

    all_rows = []
    model_wer_summary = []

    for hyp_file in hypothesis_files:
        model_id = hyp_file.stem.replace("_hypotheses", "")
        df_hyp = pd.read_csv(hyp_file)
        print(f"\nProcessing {model_id}: {len(df_hyp)} hypotheses")

        model_total_wer = 0.0
        model_count = 0

        for _, row in df_hyp.iterrows():
            sid = str(row["sample_id"])
            ref_text = str(reference_map.get(sid, ""))
            hyp_text = str(row.get("hypothesis_text", ""))
            accent = str(accent_map.get(sid, "unknown"))

            # Skip errors
            if hyp_text.startswith("[ERROR"):
                continue

            # Parse token logprobs
            token_lps_raw = row.get("token_logprobs", "[]")
            try:
                import json
                token_lps = json.loads(str(token_lps_raw)) if isinstance(token_lps_raw, str) else []
            except (json.JSONDecodeError, TypeError):
                token_lps = []

            # Compute WER for this sample
            try:
                sample_wer = jiwer.wer(_strip_punctuation(ref_text), _strip_punctuation(hyp_text))
            except Exception:
                sample_wer = 1.0

            model_total_wer += sample_wer
            model_count += 1

            # Word-level alignment
            alignment = compute_word_alignment(
                reference=ref_text,
                hypothesis=hyp_text,
                sample_id=sid,
                model_id=model_id,
                accent_background=accent,
                token_logprobs=token_lps if token_lps else None,
            )
            all_rows.extend(alignment)

        avg_wer = model_total_wer / model_count if model_count > 0 else 0.0
        model_wer_summary.append({
            "model_id": model_id,
            "samples": model_count,
            "average_wer": round(avg_wer, 4),
            "total_errors": sum(1 for r in all_rows if r["model_id"] == model_id and r["wer_op"] != "equal"),
        })
        print(f"  Average WER: {avg_wer:.4f}")

    # Write word alignment
    alignment_df = pd.DataFrame(all_rows)
    alignment_path = out_dir / "word_alignment.csv"
    alignment_df.to_csv(alignment_path, index=False)
    print(f"\nWrote {len(alignment_df)} word-level alignments to {alignment_path}")

    # Summary stats
    summary_path = out_dir / "wer_summary.csv"
    summary_df = pd.DataFrame(model_wer_summary)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote model WER summary to {summary_path}")
    print("\nWER Summary:")
    print(summary_df.to_string(index=False))

    # Per-dialect WER
    print("\nPer-dialect WER (by model):")
    for model_id in summary_df["model_id"]:
        model_rows = [r for r in all_rows if r["model_id"] == model_id]
        print(f"\n  {model_id}:")
        # Group by accent
        from collections import defaultdict
        accent_wer: dict[str, list[float]] = defaultdict(list)
        current_sid = None
        current_wer = 0.0
        current_accent = None
        error_count = 0
        word_count = 0

        # Recompute per-sample WER per accent
        for _, ref in df_samples.iterrows():
            sid = str(ref["sample_id"])
            accent = str(ref["accent_background"])
            ref_text = str(ref["text"])
            hyp_rows = df_hyp[df_hyp["sample_id"] == sid]
            if len(hyp_rows) == 0:
                continue
            hyp_text = str(hyp_rows.iloc[0].get("hypothesis_text", ""))
            if hyp_text.startswith("[ERROR"):
                continue
            try:
                w = jiwer.wer(_strip_punctuation(ref_text), _strip_punctuation(hyp_text))
            except Exception:
                w = 1.0
            accent_wer[accent].append(w)

        for accent, wers in sorted(accent_wer.items()):
            if wers:
                print(f"    {accent}: mean_wer={sum(wers)/len(wers):.4f} (n={len(wers)})")

    print("\nDone.")


if __name__ == "__main__":
    main()