#!/usr/bin/env python3
"""
compute_benchmark_metrics.py - Step 5: Post-annotation metrics computation.

After human annotation of word_alignment.csv, computes:
  - raw_wer: standard WER
  - faithful_wer: WER counting only phonetic hallucinations as errors
  - dialect_penalty: raw_wer - faithful_wer
  - hallucination_rate: % of word errors that are actual hallucinations
  - normalization_overcorrection_rate
  - Per-model, per-dialect breakdown

If the 'classification' column is missing, auto-classifies using
FAITHFUL_DIALECT_PAIRS and DIALECT_NORMALIZATION_PAIRS lookup tables.

Usage:
    python scripts/compute_benchmark_metrics.py [--config CONFIG]
    python scripts/compute_benchmark_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from accent_labs_config import REPO_ROOT, load_config, resolve_path


# (ref_word, hyp_word) where ref is standard English, hyp is Naija/Pidgin variant
FAITHFUL_DIALECT_PAIRS: set[tuple[str, str]] = {
    ("are", "dey"),
    ("are", "na"),
    ("be", "dey"),
    ("before", "bifor"),
    ("cannot", "no"),
    ("coming", "comin"),
    ("confusion", "katakata"),
    ("do", "dey"),
    ("eat", "chop"),
    ("finished", "don"),
    ("finished", "gbeda"),
    ("foreign", "jand"),
    ("get", "dey"),
    ("give", "dash"),
    ("going", "goin"),
    ("has", "dey"),
    ("help", "helep"),
    ("is", "na"),
    ("is", "dey"),
    ("just", "sha"),
    ("know", "sabi"),
    ("look", "luke"),
    ("nothing", "notin"),
    ("now", "nau"),
    ("people", "pipo"),
    ("person", "pipo"),
    ("please", "abeg"),
    ("she", "im"),
    ("small", "smol"),
    ("something", "sometin"),
    ("story", "gist"),
    ("talk", "yarn"),
    ("the", "di"),
    ("they", "dem"),
    ("thing", "tin"),
    ("things", "tins"),
    ("trouble", "wahala"),
    ("that", "dat"),
}

# (ref_word, hyp_word) where ref is Naija/Pidgin, hyp is standard English overcorrection
DIALECT_NORMALIZATION_PAIRS: set[tuple[str, str]] = {
    ("abi", "able"),
    ("am", "him"),
    ("am", "them"),
    ("bifor", "before"),
    ("bin", "been"),
    ("bin", "did"),
    ("bin", "has"),
    ("bin", "had"),
    ("bin", "was"),
    ("bin", "were"),
    ("chop", "eat"),
    ("comot", "come"),
    ("comot", "leave"),
    ("comot", "remove"),
    ("comot", "get"),
    ("dat", "that"),
    ("deir", "there"),
    ("deir", "their"),
    ("dem", "them"),
    ("dem", "they"),
    ("dem", "then"),
    ("dem", "the"),
    ("dey", "they"),
    ("dey", "are"),
    ("dey", "day"),
    ("dey", "stay"),
    ("dey", "there"),
    ("dey", "the"),
    ("di", "the"),
    ("di", "this"),
    ("di", "there"),
    ("don", "done"),
    ("don", "down"),
    ("don", "don"),
    ("e", "he"),
    ("e", "it"),
    ("e", "she"),
    ("fit", "fit"),
    ("go", "go"),
    ("im", "him"),
    ("im", "he"),
    ("im", "his"),
    ("kain", "kind"),
    ("make", "make"),
    ("mek", "make"),
    ("na", "is"),
    ("na", "are"),
    ("na", "it"),
    ("na", "and"),
    ("na", "in"),
    ("na", "of"),
    ("na", "that"),
    ("na", "the"),
    ("nau", "now"),
    ("no", "know"),
    ("notin", "nothing"),
    ("oda", "other"),
    ("pikin", "children"),
    ("pikin", "child"),
    ("pikin", "kid"),
    ("pipo", "people"),
    ("sabi", "know"),
    ("sabi", "savvy"),
    ("sey", "say"),
    ("sey", "says"),
    ("sey", "said"),
    ("sha", "share"),
    ("sometin", "something"),
    ("taym", "time"),
    ("tin", "thing"),
    ("tins", "things"),
    ("tok", "talk"),
    ("tok", "tell"),
    ("una", "you"),
    ("wahala", "trouble"),
    ("wey", "way"),
    ("wey", "where"),
    ("wey", "who"),
    ("wey", "which"),
    ("wey", "were"),
    ("wetin", "what"),
    ("wetin", "water"),
}

# Multi-word dialect variants: sorted longest-first so "are not" matches before "are"
FAITHFUL_DIALECT_MULTI_WORD: dict[str, str] = {
    "are not": "no bi",
    "cannot": "no fit",
    "they are": "dem dey",
    "he said": "im tok",
    "that thing": "dat tin",
}


def auto_classify(
    ref_word: str,
    hyp_word: str,
    wer_op: str,
) -> tuple[str, str] | None:
    """Auto-classify a word error. Returns (classification, cause) or None if equal."""
    ref_lower = ref_word.lower().strip()
    hyp_lower = hyp_word.lower().strip()

    if not ref_lower:
        if wer_op == "insert":
            return ("hallucination", "noise")
        return ("hallucination", "phonetic")

    if not hyp_lower:
        return ("hallucination", "phonetic")

    if wer_op == "substitute":
        pair = (ref_lower, hyp_lower)
        if pair in FAITHFUL_DIALECT_PAIRS:
            return ("faithful_dialect", "")
        if pair in DIALECT_NORMALIZATION_PAIRS:
            return ("hallucination", "dialect_normalization")

    if wer_op == "delete":
        return ("hallucination", "phonetic")

    if wer_op == "insert":
        return ("hallucination", "noise")

    return ("hallucination", "phonetic")


def compute_metrics(alignment_df: pd.DataFrame, unknown_pairs_log: list | None = None) -> dict:
    """Compute all benchmark metrics from the annotated word alignment.

    If 'classification' column is missing, auto-classifies using dialect pair tables.
    """
    metrics = {}

    if unknown_pairs_log is None:
        unknown_pairs_log = []

    df = alignment_df.copy()

    if "classification" not in df.columns:
        classifications = []
        causes = []
        for _, row in df.iterrows():
            result = auto_classify(
                str(row.get("ref_word", "")),
                str(row.get("hyp_word", "")),
                str(row.get("wer_op", "equal")),
            )
            if result:
                cls, cause = result
                classifications.append(cls)
                causes.append(cause)

                if cls == "hallucination" and cause == "phonetic" and row["wer_op"] == "substitute":
                    pair = (str(row["ref_word"]).lower().strip(), str(row["hyp_word"]).lower().strip())
                    if pair not in FAITHFUL_DIALECT_PAIRS and pair not in DIALECT_NORMALIZATION_PAIRS:
                        unknown_pairs_log.append({
                            "ref_word": str(row["ref_word"]),
                            "hyp_word": str(row["hyp_word"]),
                            "sample_id": str(row["sample_id"]),
                            "model_id": str(row["model_id"]),
                            "accent_background": str(row["accent_background"]),
                        })
            else:
                classifications.append("")
                causes.append("")
        df["classification"] = classifications
        df["cause"] = causes
    else:
        df["classification"] = df["classification"].fillna("")
        df["cause"] = df["cause"].fillna("")

    # Per-model, per-dialect
    for (model_id, accent), group in df.groupby(["model_id", "accent_background"]):
        total_words = len(group)
        error_rows = group[group["wer_op"] != "equal"]

        if total_words == 0:
            continue

        raw_wer = len(error_rows) / total_words

        hallucinated = error_rows[error_rows["classification"] == "hallucination"]
        faithful_wer = len(hallucinated) / total_words
        dialect_penalty = raw_wer - faithful_wer

        hallucination_rate = len(hallucinated) / len(error_rows) if len(error_rows) > 0 else 0.0

        normalization_errors = hallucinated[hallucinated["cause"] == "dialect_normalization"]
        norm_overcorrection_rate = len(normalization_errors) / len(hallucinated) if len(hallucinated) > 0 else 0.0

        faithful_dialect_rows = error_rows[error_rows["classification"] == "faithful_dialect"]

        cause_dist = hallucinated["cause"].value_counts().to_dict() if len(hallucinated) > 0 else {}

        key = f"{model_id}::{accent}"
        metrics[key] = {
            "model_id": model_id,
            "accent_background": accent,
            "total_words": int(total_words),
            "total_errors": int(len(error_rows)),
            "hallucinations": int(len(hallucinated)),
            "faithful_dialect_variants": int(len(faithful_dialect_rows)),
            "raw_wer": round(raw_wer, 4),
            "faithful_wer": round(faithful_wer, 4),
            "dialect_penalty": round(dialect_penalty, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "normalization_overcorrection_rate": round(norm_overcorrection_rate, 4),
            "cause_distribution": {k: int(v) for k, v in cause_dist.items()},
        }

    return metrics


def compute_summary(metrics: dict) -> dict:
    """Compute overall summary across models and dialects."""
    summary = {
        "models": {},
        "overall": {},
    }

    model_groups = defaultdict(list)
    for key, m in metrics.items():
        model_groups[m["model_id"]].append(m)

    for model_id, mlist in model_groups.items():
        total_words = sum(m["total_words"] for m in mlist)
        total_errors = sum(m["total_errors"] for m in mlist)
        total_hallucinations = sum(m["hallucinations"] for m in mlist)
        total_faithful = sum(m["faithful_dialect_variants"] for m in mlist)

        summary["models"][model_id] = {
            "total_words": total_words,
            "total_errors": total_errors,
            "hallucinations": total_hallucinations,
            "faithful_dialect_variants": total_faithful,
            "raw_wer": round(total_errors / total_words, 4) if total_words > 0 else 0,
            "faithful_wer": round(total_hallucinations / total_words, 4) if total_words > 0 else 0,
            "dialect_penalty": round(total_faithful / total_words, 4) if total_words > 0 else 0,
            "hallucination_rate": round(total_hallucinations / total_errors, 4) if total_errors > 0 else 0,
            "dialects": {m["accent_background"]: m["raw_wer"] for m in mlist},
            "dialect_penalties": {m["accent_background"]: m["dialect_penalty"] for m in mlist},
        }

    total_w = sum(m["total_words"] for m in metrics.values())
    total_e = sum(m["total_errors"] for m in metrics.values())
    total_h = sum(m["hallucinations"] for m in metrics.values())
    total_f = sum(m["faithful_dialect_variants"] for m in metrics.values())

    summary["overall"] = {
        "total_words": total_w,
        "total_errors": total_e,
        "hallucinations": total_h,
        "faithful_dialect_variants": total_f,
        "raw_wer": round(total_e / total_w, 4) if total_w > 0 else 0,
        "faithful_wer": round(total_h / total_w, 4) if total_w > 0 else 0,
        "dialect_penalty": round(total_f / total_w, 4) if total_w > 0 else 0,
        "hallucination_rate": round(total_h / total_e, 4) if total_e > 0 else 0,
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dialect-aware benchmark metrics.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--alignment", type=str, default=None,
                        help="Path to annotated word_alignment.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = REPO_ROOT

    align_path = resolve_path(repo_root, args.alignment or "data/kaggle_benchmark/annotations/word_alignment.csv")
    if not align_path.is_file():
        sys.exit(f"[ERROR] Alignment CSV not found: {align_path}")

    out_dir = resolve_path(repo_root, args.output_dir or "data/kaggle_benchmark/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(align_path)
    print(f"Loaded {len(df)} word alignments")

    unknown_pairs_log: list[dict] = []

    if "classification" not in df.columns:
        print("[INFO] No 'classification' column found. Running auto-classification...")
        print(f"  FAITHFUL_DIALECT_PAIRS: {len(FAITHFUL_DIALECT_PAIRS)} pairs")
        print(f"  DIALECT_NORMALIZATION_PAIRS: {len(DIALECT_NORMALIZATION_PAIRS)} pairs")

    if args.dry_run:
        total_errors = len(df[df["wer_op"] != "equal"])
        total_words = len(df)
        print(f"\n[DRY-RUN] Word alignment stats:")
        print(f"  Total words: {total_words}")
        print(f"  Total errors (non-equal): {total_errors} ({total_errors/total_words*100:.1f}%)")
        print(f"  Would write results to {out_dir / 'results.json'}")
        return

    metric_rows = compute_metrics(df, unknown_pairs_log)
    summary = compute_summary(metric_rows)

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metric_rows,
            "summary": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(metric_rows)} metric entries to {results_path}")

    csv_rows = []
    for key, m in metric_rows.items():
        m_copy = dict(m)
        m_copy["cause_distribution"] = str(m_copy.get("cause_distribution", {}))
        csv_rows.append(m_copy)
    csv_df = pd.DataFrame(csv_rows)
    csv_path = out_dir / "metrics_per_dialect.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Wrote per-dialect metrics to {csv_path}")

    if unknown_pairs_log:
        unknown_path = out_dir / "unknown_dialect_pairs.csv"
        pd.DataFrame(unknown_pairs_log).to_csv(unknown_path, index=False)
        print(f"Wrote {len(unknown_pairs_log)} unknown dialect pairs to {unknown_path}")

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    for model_id, model_data in summary["models"].items():
        print(f"\nModel: {model_id}")
        print(f"  Raw WER:  {model_data['raw_wer']:.4f}")
        print(f"  Faithful WER: {model_data['faithful_wer']:.4f}")
        print(f"  Dialect Penalty: {model_data['dialect_penalty']:.4f}")
        print(f"  Hallucination Rate: {model_data['hallucination_rate']:.2%}")
        print(f"  Errors: {model_data['total_errors']} ({model_data['faithful_dialect_variants']} faithful + {model_data['hallucinations']} hallucination)")
        print("\n  Per-dialect WER:")
        for dialect, wer in sorted(model_data["dialects"].items()):
            penalty = model_data["dialect_penalties"].get(dialect, 0)
            mark = " (low)" if penalty < 0.02 else ""
            print(f"    {dialect}: raw_wer={wer:.4f} penalty={penalty:.4f}{mark}")
        print(f"\n  Per-dialect dialect_penalty:")
        for dialect, penalty in sorted(model_data["dialect_penalties"].items(), key=lambda x: -x[1]):
            print(f"    {dialect}: {penalty:.4f}")

    print(f"\nOverall:")
    for k, v in summary["overall"].items():
        print(f"  {k}: {v}" if not isinstance(v, float) else f"  {k}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()