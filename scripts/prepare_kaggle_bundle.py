#!/usr/bin/env python3
"""
prepare_kaggle_bundle.py - Package 500 audio files for Kaggle private dataset upload.

Creates data/kaggle_benchmark/kaggle_bundle/:
  audio/                          — 500 FLAC files
  test_samples.csv                — clean metadata (no local paths)

Usage:
    python scripts/prepare_kaggle_bundle.py [--config CONFIG]
    python scripts/prepare_kaggle_bundle.py --dry-run

After running, upload the kaggle_bundle/ directory as a new Kaggle private dataset.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from accent_labs_config import REPO_ROOT, load_config, resolve_path

# Columns to keep in the Kaggle metadata CSV (no local paths)
KAGGLE_METADATA_COLUMNS = [
    "sample_id", "text", "speaker_id", "accent_background",
    "ethnolinguistic_group", "register", "duration_s",
    "snr_estimate", "noise_type", "language",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kaggle bundle for inference.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--samples", type=str, default=None,
                        help="Path to test_samples.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/kaggle_benchmark/kaggle_bundle)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = REPO_ROOT

    out_dir = resolve_path(repo_root, args.output_dir or "data/kaggle_benchmark/kaggle_bundle")
    audio_dir = out_dir / "audio"

    samples_path = resolve_path(repo_root, args.samples or "data/kaggle_benchmark/dataset/test_samples.csv")
    if not samples_path.is_file():
        sys.exit(f"[ERROR] Samples CSV not found: {samples_path}")

    df = pd.read_csv(samples_path)
    print(f"Loaded {len(df)} samples from {samples_path}")

    if args.dry_run:
        print(f"\n[DRY-RUN] Would create bundle at {out_dir}:")
        print(f"  Copy {len(df)} audio files to {audio_dir}/")
        print(f"  Write clean metadata CSV to {out_dir / 'test_samples.csv'}")
        print(f"  Columns: {KAGGLE_METADATA_COLUMNS}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio files, using audio_path column (resolved by select_benchmark_samples.py)
    missing_files = 0
    audio_col = "audio_path" if "audio_path" in df.columns else "audio"
    for _, row in df.iterrows():
        src = row.get(audio_col, "")
        if not src or pd.isna(src):
            missing_files += 1
            continue
        src_path = Path(str(src))
        if not src_path.is_file():
            missing_files += 1
            continue
        shutil.copy2(str(src_path), audio_dir / src_path.name)

    print(f"Copied {len(df) - missing_files} audio files to {audio_dir}/")
    if missing_files:
        print(f"  [WARN] {missing_files} audio files not found, skipped")

    # Write clean metadata CSV (no local machine paths)
    available_cols = [c for c in KAGGLE_METADATA_COLUMNS if c in df.columns]
    meta_df = df[available_cols].copy()
    csv_path = out_dir / "test_samples.csv"
    meta_df.to_csv(csv_path, index=False)
    print(f"Wrote metadata ({len(available_cols)} columns) to {csv_path}")

    # Size summary
    total_size = sum(f.stat().st_size for f in audio_dir.iterdir() if f.is_file())
    print(f"\nTotal audio size: {total_size / 1e9:.2f} GB ({len(df) - missing_files} files)")

    print("\n" + "=" * 60)
    print("UPLOAD INSTRUCTIONS")
    print("=" * 60)
    print(f"1. Zip the bundle:  cd {out_dir.parent} && tar czf kaggle_bundle.tar.gz kaggle_bundle/")
    print(f"   (On Windows: zip -r kaggle_bundle.zip kaggle_bundle/)")
    print(f"\n2. Go to https://www.kaggle.com/datasets and click 'New Dataset'")
    print(f"3. Upload the zip file")
    print(f"4. Set title: 'Naija ASR Benchmark — Audio'")
    print(f"5. Set license: CC-BY-4.0")
    print(f"6. Keep it PRIVATE (you'll delete it after inference)")
    print(f"7. Note the dataset name (e.g., 'your-username/naija-asr-benchmark-audio')")
    print(f"\n8. In the inference notebook, set DATASET_NAME to the Kaggle dataset name")
    print(f"9. Add HUGGINGFACE_HUB_TOKEN as a Kaggle Secret in notebook settings")
    print(f"10. Run with GPU accelerator (T4 or P100)")
    print(f"\n11. Also upload notebooks/kaggle_benchmark_inference.ipynb to Kaggle as a notebook")
    print(f"12. Delete the private audio dataset from Kaggle")
    print("\nDone.")


if __name__ == "__main__":
    main()