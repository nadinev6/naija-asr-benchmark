#!/usr/bin/env python3
"""
Create Kaggle-safe version of test_samples.csv (strip audio_path and local path columns).

Usage:
    python scripts/create_kaggle_csv.py [--config CONFIG]
    python scripts/create_kaggle_csv.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from accent_labs_config import REPO_ROOT, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Kaggle-safe test_samples CSV.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input", type=str, default=None,
                        help="Path to source test_samples.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/kaggle_benchmark/dataset)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = REPO_ROOT

    src_path = resolve_path(repo_root, args.input or "data/kaggle_benchmark/dataset/test_samples.csv")
    if not src_path.is_file():
        sys.exit(f"[ERROR] Source CSV not found: {src_path}")

    out_dir = resolve_path(repo_root, args.output_dir or "data/kaggle_benchmark/dataset")
    out_path = out_dir / "test_samples_kaggle.csv"

    df = pd.read_csv(src_path)
    print(f"Loaded {len(df)} rows from {src_path}")
    print(f"Columns: {list(df.columns)}")

    # Columns to strip that reveal local machine structure
    strip_cols = ["audio_path", "audio", "_source", "created_at"]
    existing_strip = [c for c in strip_cols if c in df.columns]
    if existing_strip:
        df = df.drop(columns=existing_strip)
        print(f"Dropped columns: {existing_strip}")

    if args.dry_run:
        print(f"\n[DRY-RUN] Would write {len(df)} rows to {out_path}")
        print(f"  Columns: {list(df.columns)}")
        return

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()