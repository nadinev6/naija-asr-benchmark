#!/usr/bin/env python3
"""
run_benchmark_inference.py - Step 2: Run ASR inference for all 3 benchmark models.

Models:
  1. openai/whisper-large-v3       — zero-shot general-purpose baseline
  2. accent-labs-finetuned          — Accent Labs fine-tuned Whisper (from config model_dir)
  3. openai/whisper-small           — lightweight baseline for dialect-gap comparison

Output: data/kaggle_benchmark/model_outputs/{model_id}_hypotheses.csv

Usage:
    python scripts/run_benchmark_inference.py [--config CONFIG] [--samples CSV] [--models MODEL1 MODEL2 ...]
    python scripts/run_benchmark_inference.py --dry-run
    python scripts/run_benchmark_inference.py --models whisper-small  # run single model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from accent_labs_config import REPO_ROOT, load_config, resolve_path
from whisper_asr_utils import load_audio_mono, load_whisper_model, transcribe_one_with_confidence


DEFAULT_MODELS = {
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-small": "openai/whisper-small",
    "naija-finetuned": "nadinev/naija-voice-model",
}


def resolve_model_path(model_id: str, model_name_or_path: str, cfg: dict, hf_token: str | None = None) -> str:
    """Resolve model path or HF Hub ID, handling auth tokens."""
    if model_id == "accent-labs-finetuned":
        paths_cfg = cfg.get("paths", {})
        return str(resolve_path(REPO_ROOT, paths_cfg.get("model_dir", "/workspace/whisper-lora-merged")))
    if model_id == "naija-finetuned" and hf_token:
        return model_name_or_path  # HF Hub ID, token handled by load_whisper_model
    return model_name_or_path


@torch.inference_mode()
def transcribe_with_word_probs(
    model,
    processor,
    audio: np.ndarray,
    *,
    device: str,
    use_bf16: bool,
    language: str,
    task: str,
    max_new_tokens: int = 225,
    num_beams: int = 10,
    no_repeat_ngram_size: int = 3,
) -> dict:
    """Transcribe audio and return text, per-token log probs, and avg log prob."""
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if use_bf16:
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    elif device == "cuda":
        inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        language=language,
        task=task,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=1.0,
        condition_on_prev_tokens=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    token_ids = outputs.sequences[0].tolist()
    all_logprobs = []
    all_token_ids = []
    if outputs.scores:
        for i, score in enumerate(outputs.scores):
            probs = torch.nn.functional.log_softmax(score, dim=-1)
            token_id = outputs.sequences[0, i + 1].item() if i + 1 < outputs.sequences.shape[1] else None
            if token_id is not None:
                all_logprobs.append(float(probs[0, token_id].item()))
                all_token_ids.append(token_id)

    avg_logprob = float(np.mean(all_logprobs)) if all_logprobs else 0.0
    min_logprob = float(np.min(all_logprobs)) if all_logprobs else 0.0

    return {
        "text": text,
        "avg_logprob": avg_logprob,
        "min_logprob": min_logprob,
        "token_logprobs": json.dumps(all_logprobs),
        "token_ids": json.dumps(all_token_ids),
        "n_tokens": len(all_logprobs),
    }


def run_model_inference(
    model_id: str,
    model_name_or_path: str,
    samples_df: pd.DataFrame,
    output_path: Path,
    cfg: dict,
    device: str,
    dry_run: bool = False,
    hf_token: str | None = None,
) -> pd.DataFrame:
    """Run inference for one model on all samples. Returns results DataFrame."""
    paths_cfg = cfg.get("paths", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    language = str(train_cfg.get("language", "en"))
    task_type = str(train_cfg.get("task", "transcribe"))
    max_new = int(eval_cfg.get("generation_max_length", 225))
    sr = int(train_cfg.get("whisper_sample_rate", 16000))
    batch_size = int(eval_cfg.get("batch_size", 4))

    # Resolve model path for local fine-tuned or HF Hub models
    model_path = resolve_model_path(model_id, model_name_or_path, cfg, hf_token)

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")
    print(f"  Samples: {len(samples_df)}")

    if dry_run:
        print(f"  [DRY-RUN] Would load model and run inference")
        # Return empty placeholder
        return pd.DataFrame(columns=[
            "sample_id", "model_id", "hypothesis_text", "avg_logprob",
            "min_logprob", "token_logprobs", "token_ids", "n_tokens",
            "inference_time_s", "accent_background"
        ])

    # Load model with auth token for HF Hub models
    use_hf_auth = hf_token is not None and model_id == "naija-finetuned"
    model, processor, use_bf16 = load_whisper_model(model_path, device, hf_token=hf_token if use_hf_auth else None)

    rows = []
    start_time = time.time()

    for idx, row in samples_df.iterrows():
        audio_path = row.get("audio_path", row.get("audio", ""))
        sample_id = str(row.get("sample_id", f"sample_{idx}"))
        accent = str(row.get("accent_background", "unknown"))

        if not audio_path or not Path(str(audio_path)).is_file():
            print(f"  [WARN] Audio not found: {audio_path}")
            rows.append({
                "sample_id": sample_id,
                "model_id": model_id,
                "hypothesis_text": "",
                "avg_logprob": 0.0,
                "min_logprob": 0.0,
                "token_logprobs": "[]",
                "token_ids": "[]",
                "n_tokens": 0,
                "inference_time_s": 0.0,
                "accent_background": accent,
            })
            continue

        try:
            audio = load_audio_mono(str(audio_path), sr)
            sample_start = time.time()
            result = transcribe_with_word_probs(
                model, processor, audio,
                device=device, use_bf16=use_bf16,
                language=language, task=task_type,
                max_new_tokens=max_new,
            )
            elapsed = time.time() - sample_start
            rows.append({
                "sample_id": sample_id,
                "model_id": model_id,
                "hypothesis_text": result["text"],
                "avg_logprob": result["avg_logprob"],
                "min_logprob": result["min_logprob"],
                "token_logprobs": result["token_logprobs"],
                "token_ids": result["token_ids"],
                "n_tokens": result["n_tokens"],
                "inference_time_s": round(elapsed, 3),
                "accent_background": accent,
            })
        except Exception as e:
            print(f"  [ERROR] Failed {sample_id}: {e}")
            rows.append({
                "sample_id": sample_id,
                "model_id": model_id,
                "hypothesis_text": f"[ERROR: {e}]",
                "avg_logprob": 0.0,
                "min_logprob": 0.0,
                "token_logprobs": "[]",
                "token_ids": "[]",
                "n_tokens": 0,
                "inference_time_s": 0.0,
                "accent_background": accent,
            })

        if (idx + 1) % 50 == 0:
            elapsed_total = time.time() - start_time
            rate = (idx + 1) / elapsed_total if elapsed_total > 0 else 0
            print(f"  Progress: {idx + 1}/{len(samples_df)} ({rate:.1f} samples/s)")

    total_time = time.time() - start_time
    print(f"  Completed {len(rows)} samples in {total_time:.1f}s ({len(rows)/total_time:.1f} samples/s)")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"  Wrote {output_path}")

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark model inference.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--samples", type=str, default=None,
                        help="Override path to test_samples.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override model outputs directory")
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS.keys()),
                        help=f"Models to run (default: {' '.join(DEFAULT_MODELS.keys())})")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace Hub auth token for private models")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = REPO_ROOT

    samples_path = resolve_path(repo_root, args.samples or "data/kaggle_benchmark/dataset/test_samples.csv")
    if not samples_path.is_file():
        sys.exit(f"[ERROR] Samples CSV not found: {samples_path}\nRun scripts/select_benchmark_samples.py first.")

    out_dir = resolve_path(repo_root, args.output_dir or "data/kaggle_benchmark/model_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    df_samples = pd.read_csv(samples_path)
    print(f"Loaded {len(df_samples)} test samples from {samples_path}")

    # Build model map from args
    model_map = {}
    for m in args.models:
        if m in DEFAULT_MODELS:
            model_map[m] = DEFAULT_MODELS[m]
        elif m == "accent-labs-finetuned":
            model_map[m] = "local"  # resolved at runtime
        else:
            model_map[m] = m  # treat as HuggingFace model ID or local path

    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
    if hf_token:
        print(f"HuggingFace Hub auth: token provided ({hf_token[:8]}...)")

    print(f"\nModels to run: {list(model_map.keys())}")

    if args.dry_run:
        print("\n[DRY-RUN] Would run:")
        for model_id, model_path in model_map.items():
            out_path = out_dir / f"{model_id}_hypotheses.csv"
            print(f"  {model_id}: {model_path} -> {out_path}")
        return

    # Run each model
    all_results = {}
    for model_id, model_name_or_path in model_map.items():
        out_path = out_dir / f"{model_id}_hypotheses.csv"
        if out_path.is_file():
            print(f"\n[SKIP] {model_id} already exists at {out_path}")
            all_results[model_id] = pd.read_csv(out_path)
            continue

        result_df = run_model_inference(
            model_id, model_name_or_path,
            df_samples, out_path, cfg, device,
            dry_run=False,
            hf_token=hf_token,
        )
        all_results[model_id] = result_df

    print("\nDone. All model outputs written.")


if __name__ == "__main__":
    main()