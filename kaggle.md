*This is a submission for the [Kaggle Benchmarking Challenge](https://dev.to/challenges/kaggle-2026-09-23)*

## TL;DR

- Fine-tuning Whisper large-v3 on Nigerian English / Pidgin cut WER from **35.05% to 8.30%** (a 76% relative reduction), though the test samples may overlap the fine-tuning data — treat this as an upper bound on the improvement.
- Standard WER turned out to be **nearly fair** on this data, though the fine-tuned figure may be optimistic. The dialect-spelling penalty was only **0.30 points** for the fine-tuned model and 0.00 for the baselines, because the baselines never write dialect spellings like "pipo".
- The bigger problem is **overcorrection**: the baselines silently "fix" Pidgin words (e.g. "dey" → "they"), which makes up 18.5-23.5% of their errors vs 10.2% for the fine-tuned model.

## What I Benchmarked

I built a benchmark to separate real speech errors from correct local dialect spellings in Nigerian English and Pidgin.

Standard automatic speech recognition (ASR) accuracy tests mark a model wrong when it correctly transcribes dialect words, slang, or Pidgin. Most evaluation relies on word error rate (WER) computed against references written in standard English spelling. If a model transcribes "pipo" (what the speaker said) but the reference says "people", standard WER counts a substitution error. That penalizes a spelling difference, not a listening mistake.

The benchmark separates real recognition errors from dialect-spelling mismatches and reports four metrics:

- **Raw WER:** standard WER.
- **Faithful WER:** WER after discounting faithful dialect variants (spellings like "pipo" that match what was spoken).
- **Dialect penalty:** Raw WER minus Faithful WER, in percentage points.
- **Normalization overcorrection:** the share of errors where the reference has a dialect word and the model "corrects" it to standard English.

I selected 500 samples across 5 speaker backgrounds (Igbo, Yoruba, Hausa, Pidgin, multilingual), ran 3 Whisper models on them, and built an auto-classifier with 120 known dialect substitution rules to compute dialect-aware metrics without waiting for full human annotation.

## Models Tested

| Model | Parameters | Role |
|-------|-----------|------|
| **OpenAI Whisper large-v3** | 1,540M | Zero-shot general-purpose baseline |
| **OpenAI Whisper small** | 244M | Lightweight baseline for dialect-gap comparison |
| **Accent Labs fine-tuned Whisper** | 1,540M | My fine-tune of large-v3 (`nadinev/naija-voice-model`) |

I chose Whisper because it's among the most widely deployed ASR pipelines in production, and comparing zero-shot and fine-tuned models on the same architecture shows how much dialect-specific training helps.

The fine-tuned model was trained on ~24K Nigerian English / Pidgin samples with accent-weighted sampling. Inference for all 3 models ran on Kaggle's free GPU.

- **Test data:** The 500 test samples were drawn from the same combined manifest used for fine-tuning, so they may overlap with the fine-tuned model's training data. The zero-shot baselines never saw this data.
- **Inference settings:** Language was forced to `en` (not auto-detected), task was `transcribe`. Punctuation is stripped and text is lowercased before WER alignment. Generation used `num_beams=10`, `no_repeat_ngram_size=3`, `max_new_tokens=225`.

**Punctuation artifact:** The original run didn't strip punctuation before WER alignment. Whisper outputs capitalization and punctuation by default, so trailing periods and commas inflated the baselines — "people.," counted as an error against "people". After fixing this, large-v3 dropped from 41.92% to 35.05%, small from 48.00% to 41.19%, and the fine-tuned model went from 11.11% to 8.30%. The gap narrowed from 30.8 to 26.8 percentage points, but the overcorrection finding — the article's main insight — is unaffected. Raw numbers are preserved in `results_original.json`.

## Auto-Classifier Design

Rather than manually labeling roughly 6,700 word errors across the three models (645 for the fine-tuned model alone), I built a rule-based auto-classifier in `compute_benchmark_metrics.py`. It labels two error types from word-level alignment:

1. **FAITHFUL_DIALECT_PAIRS** (38 pairs): `(ref_word, hyp_word)` where the reference is standard English and the hypothesis is a known Naija/Pidgin variant.
   These are **not** real errors and are discounted in Faithful WER.

2. **DIALECT_NORMALIZATION_PAIRS** (82 pairs): `(ref_word, hyp_word)` where the reference contains a Naija/Pidgin word and the hypothesis over-corrects to standard English.
   These **are** errors, caused by the model "fixing" dialect.

| Reference | Model output | Class | Real error? |
|-----------|--------------|-------|-------------|
| people | pipo | Faithful variant | No |
| the | di | Faithful variant | No |
| dey | they | Overcorrection | Yes |
| dem | them | Overcorrection | Yes |
| na | is | Overcorrection | Yes |

Substitutions that match neither list default to `phonetic` and are logged to `unknown_dialect_pairs.csv` for review.

## Benchmark Results

| Metric | Whisper large-v3 | Whisper small | Fine-tuned |
|--------|---------:|--------:|--------:|
| Raw WER | 35.05% | 41.19% | 8.30% |
| Faithful WER | 35.05% | 41.19% | 8.00% |
| Dialect penalty (pts) | 0.00 | 0.00 | 0.30 |
| Normalization overcorrection (% of errors) | 23.5% | 18.5% | 10.2% |

*Fine-tuned WER on the 160 held-out (non-overlapping) samples: 7.11%.

**Per-dialect raw WER:**

| Dialect | Whisper large-v3 | Whisper small | Fine-tuned |
|---------|---------:|--------:|--------:|
| Hausa | 35.41% | 41.86% | 9.81% |
| Igbo | 35.38% | 42.92% | 7.75% |
| Multilingual | 27.65% | 33.69% | 3.84% |
| Pidgin | 35.02% | 40.23% | 7.46% |
| Yoruba | 36.51% | 41.49% | 9.34% |

**Cause distribution (auto-classified, % of word errors):**

- **Phonetic:** acoustic confusion — the model substituted a wrong word, deleted a word, or produced output not matching the audio signal. Also the default catch-all for any substitution that doesn't match a known dialect pair.
- **Dialect normalization:** the model replaced a dialect word with its standard-English equivalent.
- **Noise:** inserted words with no counterpart in the reference transcript.

| Model | Phonetic | Dialect Normalization | Noise |
|-------|---------:|---------------------:|-----:|
| Whisper large-v3 | 71.1% | 23.5% | 5.4% |
| Whisper small | 75.8% | 18.5% | 5.7% |
| Fine-tuned | 81.0% | 10.6% | 8.4% |

### What the results show

- **Fine-tuning helps, but the gap is an upper bound.** WER fell by 76% relative (35.05% → 8.30%), though the test samples may overlap the fine-tuning data. The zero-shot models struggled on Nigerian English, with 28-43% WER across all accents, likely because this speech sits far from the standard English that dominates Whisper's training data.
- **Overcorrection drops from 23.5% to 10.2%.** The fine-tuned model is much less aggressive about "correcting" dialect words like "dey", "dem", and "na" into standard English.
- **Faithful dialect variants appear only in the fine-tuned model** (23 of 645 word errors). The zero-shot models never produce forms like "pipo" or "wahala" because they always normalize to standard English, so FAITHFUL_DIALECT_PAIRS never match.
- **The dialect penalty is small:** 0.30 points overall for the fine-tuned model. It's highest for Igbo (0.43) and Yoruba (0.42), but with only 23 such events, that's too few to rank dialects.
- **Multilingual speech has the lowest WER across all models,** possibly because code-switched speech contains more standard English.
- **Hausa and Yoruba have the highest WER** even for the fine-tuned model (9.81% and 9.34%), which suggests these accents could benefit from more training data.

## Limitations

- The classifier uses 120 hand-curated dialect pair rules and hasn't been validated against human annotation. An annotation guide and blind-review workflow exist but haven't been run on the benchmark results. Anything outside the 120 pairs defaults to `phonetic`, so the dialect-aware metrics are lower bounds.
- The test samples may overlap the fine-tuned model's training data, so the fine-tuned results are an upper bound on the improvement.
- With 500 samples across 5 groups, per-dialect numbers come from small samples and have no confidence intervals.
- Audio files are private, so the full pipeline can't be re-run end to end. Hypothesis outputs and alignments are included.

## What's Next

Expand the rule set using the pairs logged in `unknown_dialect_pairs.csv`, and validate the classifier with human annotation using the included annotation guide.

## Reproduce It

The benchmark and inference notebook are on Kaggle: [Naija ASR Benchmark Inference Notebook](https://www.kaggle.com/code/nadinev6/naija-asr-benchmark)

**Data**
- Hypothesis outputs from Whisper large-v3, Whisper small, and the fine-tuned model
- Word-level alignment with token log probabilities and auto-classification
- 120 rule-based dialect pair tables for error classification
- Logged unknown dialect pairs for expanding the rule set
- Annotation guide for human error classification

**Code and notebooks**
- Analysis scripts: `compute_benchmark_wer.py`, `compute_benchmark_metrics.py`
- Notebook for per-dialect WER, penalty, and cause distribution
- Kaggle inference notebook for running transcription on the free GPU

**Repository:** [github.com/accent-labs/dialect-aware-asr-benchmark](https://github.com/accent-labs/dialect-aware-asr-benchmark)