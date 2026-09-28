*This is a submission for the [Kaggle Benchmarking Challenge](https://dev.to/challenges/kaggle-2026-09-23)*

## What I Benchmarked

We benchmarked **Dialect-Aware Error Taxonomy** for Automatic Speech Recognition (ASR) on Nigerian English and Pidgin — a task that measures whether standard Word Error Rate (WER) overestimates real ASR failures by penalizing faithful dialect transcription.

Most ASR evaluation relies on standardized WER against General American or Received Pronunciation benchmarks. When speech models encounter regional dialects, standard evaluation masks a core failure mode: **lexical substitution penalty vs. phonetic hallucination**. If a model transcribes "pipo" (what the speaker said) but the reference says "people", standard WER counts a substitution error — penalizing the model for orthographic variance, not accuracy.

Our benchmark introduces a two-class error taxonomy:
- **Faithful dialect variant**: The model correctly transcribed what was spoken, but the reference uses a different orthographic or lexical convention (e.g., "dey" → "they", "wahala" → "trouble", "sabi" → "know")
- **Phonetic hallucination**: Genuine ASR failure — the output doesn't match the acoustic signal

We built a 500-sample stratified benchmark across 5 ethnolinguistic backgrounds (Igbo, Yoruba, Hausa, Pidgin, multilingual), set up 3 Whisper models for inference, and developed an auto-classifier that uses 120 known dialect substitution rules to compute dialect-aware metrics without waiting for full human annotation.

## Models Tested

| Model | Parameters | Role |
|-------|-----------|------|
| **OpenAI Whisper large-v3** | 1,540M | Zero-shot general-purpose baseline |
| **OpenAI Whisper small** | 244M | Lightweight baseline for dialect-gap comparison |
| **Accent Labs fine-tuned Whisper** | 1,540M | Domain-adapted model trained on Nigerian English speech |

We chose Whisper variants because they represent the most widely deployed ASR pipeline in production, and comparing zero-shot vs. fine-tuned performance on the same architecture isolates the effect of dialect-aware training.

The fine-tuned model (`nadinev/naija-voice-model`) was trained on ~24K Nigerian English / Pidgin samples with accent-weighted sampling. Inference was run on Kaggle free GPU using all 3 models.

## Auto-Classifier Design

Rather than requiring 7,500+ human annotations, we built a rule-based auto-classifier in `compute_benchmark_metrics.py` that identifies two error types from word-level alignment:

1. **FAITHFUL_DIALECT_PAIRS** (38 pairs): `(ref_word, hyp_word)` where the reference is standard English and the hypothesis is a known Naija/Pidgin variant — e.g., `("people", "pipo")`, `("the", "di")`, `("trouble", "wahala")`. These are NOT real errors.

2. **DIALECT_NORMALIZATION_PAIRS** (82 pairs): `(ref_word, hyp_word)` where the reference contains a Naija/Pidgin word and the hypothesis over-corrects to standard English — e.g., `("dey", "they")`, `("dem", "them")`, `("na", "is")`. These ARE hallucinations caused by the model "fixing" dialect.

Unmatched substitutions are classified as phonetic hallucinations and logged to `unknown_dialect_pairs.csv` for review.

## Benchmark Results

| Metric | Whisper large-v3 | Whisper small | Accent Labs fine-tuned |
|--------|---------:|--------:|--------:|
| Raw WER | 41.92% | 48.00% | 11.11% |
| Faithful WER | 41.92% | 48.00% | 10.81% |
| Dialect Penalty | 0.00% | 0.00% | 0.30% |
| Hallucination Rate | 100.0% | 100.0% | 97.34% |
| Normalization Overcorrection | 18.5% | 14.7% | 7.7% |

**Per-dialect raw WER:**

| Dialect | Whisper large-v3 | Whisper small | Accent Labs fine-tuned |
|---------|---------:|--------:|--------:|
| Hausa | 42.93% | 49.84% | 12.45% |
| Igbo | 42.18% | 49.81% | 10.46% |
| Multilingual | 36.49% | 41.47% | 6.92% |
| Pidgin | 40.33% | 45.30% | 9.39% |
| Yoruba | 43.37% | 48.05% | 12.99% |

**Cause distribution (auto-classified):**

| Model | Phonetic | Dialect Normalization | Noise |
|-------|---------:|---------------------:|-----:|
| Whisper large-v3 | 77.1% | 18.5% | 4.4% |
| Whisper small | 80.5% | 14.7% | 4.8% |
| Accent Labs fine-tuned | 86.6% | 7.7% | 5.7% |

**Key findings:**
- **Zero-shot models perform poorly** on Nigerian English — 37-50% WER across all accents. Accented speech is far from the standard American/British English Whisper was trained on.
- **Fine-tuning reduces WER by ~75% relative** (41.92% → 11.11% overall), demonstrating that domain adaptation on dialect-rich data dramatically improves ASR performance.
- **Dialect normalization overcorrection drops from 18.5% to 7.7%** — the fine-tuned model is less aggressive about "correcting" dialect words like "dey", "dem", and "na" to their standard English equivalents.
- **Faithful dialect variants are only detected for the fine-tuned model** (23 of 866 word errors). Zero-shot models never produce dialect forms like "pipo" or "wahala" — they always normalize to standard English, so the auto-classifier's FAITHFUL_DIALECT_PAIRS never match.
- **Dialect penalty is small but non-zero for the fine-tuned model** (0.30% overall). The penalty is highest for Igbo (0.43%) and Yoruba (0.42%), suggesting these accents trigger the most dialect variants in the fine-tuned model's output.
- **Multilingual accent has the lowest WER** across all models, likely because code-switched speech contains more standard English.
- **Hausa and Yoruba have the highest WER** even for the fine-tuned model (12.45% and 12.99%), suggesting these accents may benefit from additional training data.

## Kaggle Inference Workflow

1. ✅ **Prepare bundle** — `python scripts/prepare_kaggle_bundle.py` (completed)
2. ✅ **Upload to Kaggle** — private dataset created (completed)
3. ✅ **Set HF token** — `HUGGINGFACE_HUB_TOKEN` configured in notebook secrets (completed)
4. ✅ **Run inference** — all 3 models transcribed on Kaggle GPU (~8 hours) (completed)
5. **Download CSVs**: Save the 3 hypothesis CSVs to `data/kaggle_benchmark/model_outputs/`
6. **Delete private dataset**: Remove the audio dataset from Kaggle (no audio in the public submission)
7. ✅ **Compute metrics** — `results.json` populated with auto-classified metrics (completed)
8. **Render visualizations**: Open `notebooks/kaggle_benchmark_viz.ipynb` — per-dialect WER, dialect penalty, hallucination rate charts

## My Benchmark

The benchmark dataset is available on Kaggle:
[Naija ASR Benchmark Inference Notebook](https://www.kaggle.com/code/nadinev6/notebook6e01805d52)

The dataset includes:
- 500 test samples with reference transcripts and metadata (Kaggle-safe, no audio paths)
- Hypothesis outputs from Whisper large-v3, Whisper small, and Accent Labs fine-tuned model
- Word-level alignment with token log probabilities and auto-classification
- 120 rule-based dialect pair tables for error classification
- Annotation guide for human error classification
- Analysis scripts (compute_benchmark_wer.py, compute_benchmark_metrics.py)
- Visualization notebook for per-dialect WER, penalty, and cause distribution
- Kaggle inference notebook for running model transcription on free GPU
- Logged unknown dialect pairs for expanding the rule set

**Repository**: [github.com/nadinev6/naija-asr-benchmark](https://github.com/nadinev6/naija-asr-benchmark)