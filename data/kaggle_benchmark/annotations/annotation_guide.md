# Annotation Guide — Dialect-Aware Error Taxonomy

## Goal

Classify every word-level ASR error into one of two categories:

1. **Faithful dialect variant** — The model transcribed the audio correctly, but the reference transcript uses a different orthographic convention. The model was right; the metric penalizes it unfairly.
2. **Phonetic hallucination** — The model produced output that does not match the acoustic signal. True ASR failure.

If hallucination, also assign a **cause label**.

---

## Primary Classification

### Faithful Dialect Variant (label: `faithful_dialect`)

The hypothesis word captures what the speaker said, but the reference transcript uses a different spelling, register, or lexical convention.

| Reference | Hypothesis | Reason |
|-----------|-----------|--------|
| people | pipo | Pidgin orthographic convention |
| they | dem | Pidgin pronoun |
| are | dey | Pidgin copula (same meaning, different form) |
| she | im | Gender-neutral pronoun in Naija |
| is | na | Copula substitution in Pidgin |
| just | sha | Discourse marker preserved |
| please | abeg | Lexical borrowing, same intent |
| trouble | wahala | Arabic-origin loan word, standard in Naija |
| know | sabi | Lexical substitution, correct |
| thing | tin | Phonetic reduction, standard in Pidgin |
| give | dash | Correct colloquial form |
| eat | chop | Correct colloquial form |
| talk | yarn | Correct colloquial form |
| foreign | jand | Correct slang |
| story | gist | Correct colloquial form |
| confusion | katakata | Correct colloquial form |
| finished | gbeda | Correct colloquial form |
| finished | don | Aspectual marker for completed action |
| are not | no bi | Negation structure in Pidgin |
| they are | dem dey | Subject pronoun + copula, correct |
| he said | im tok | Correct |
| that thing | dat tin | Phonetic reduction, correct |
| small | smol | Phonetic reduction, correct |
| now | nau | Vowel shift, correct in context |
| before | bifor | Vowel reduction, correct |
| nothing | notin | Consonant cluster simplification |
| look | luke | Vowel shift, correct in Pidgin |
| help | helep | Vowel insertion, correct in Pidgin |
| something | sometin | Consonant reduction |
| coming | comin | G-dropping, standard spoken English variant |
| going | goin | G-dropping |
| inside | inside | — |
| outside | outside | — |
| can not | no fit | Pidgin negation structure |

**When in doubt:** If the hypothesis is a known Naija/Pidgin variant of an English word and the audio clearly contains the variant and not the standard form, classify as `faithful_dialect`.

### Phonetic Hallucination (label: `hallucination`)

The model output deviates from the audio signal. These are genuine ASR failures:

- Wrong word inserted that wasn't spoken
- Word dropped that was clearly audible
- Word substituted with something phonetically unrelated
- Nonsense or gibberish generated
- Stuttering artifacts ("ten ten ten", "the the the")

Examples:
| Reference | Hypothesis | Why |
|-----------|-----------|-----|
| caterpillar | caterpillar inside caterpillar | Stuttering hallucination |
| economy | icconomy | Added syllable, not in audio |
| they said | di said | Wrong pronoun |
| healthcare | headwater | Phonetically wrong |
| shortage | shorttage | Nonstandard suffix, error |
| moisture | mositure | Transposition error |
| government | goverment | Dropped syllable that was audible |

---

## Secondary Classification — Cause Labels

Only for `hallucination` errors:

| Cause | Meaning | Example |
|-------|---------|---------|
| `phonetic` | Acoustic confusion — similar-sounding word substituted | "money" → "monino" (audible nasal confusion) |
| `dialect_normalization` | Model over-corrects dialect to standard English | "dey" → "they" when audio clearly says "dey" |
| `code_switch` | Error at or near a language boundary | "I went to the market ba" → "I went to the market bar" (Yoruba utterance "ba" normalized to English "bar") |
| `noise` | Background noise, channel distortion, or clipping caused misrecognition | Reference "meeting" → hypothesis "meaning" in high-SNR clip |

---

## Annotation Workflow

1. **Open** `word_alignment.csv` — each row is one reference word with the corresponding hypothesis word (or blank for deletions)
2. **Listen** to the corresponding audio file (sample_id matches the filename)
3. **Compare** reference vs hypothesis for that word in context
4. **Fill** `classification` column: `faithful_dialect` or `hallucination`
5. **If** `hallucination`: fill `cause` column with one of `phonetic`, `dialect_normalization`, `code_switch`, `noise`

### Priority

Not every word needs annotation. Prioritize:
- **High priority**: All `substitute` rows (both words differ) — these tell us if the penalty is fair or unfair
- **Medium priority**: `delete` rows — was this word actually missing?
- **Low priority**: `insert` rows — was this a hallucinated addition?

---

## Edge Cases

| Scenario | Classification | Rationale |
|----------|---------------|-----------|
| Partial match ("shortage" → "short-") | hallucination (phonetic) | Incomplete word output |
| Disfluency preserved ("I-I-I said") | faithful_dialect | Speaker disfluency; model correctly transcribed what was said |
| Punctuation difference ("cannot" → "can not") | faithful_dialect | Orthographic variant, same lexical content |
| Capitalization only | faithful_dialect | Cosmetic, zero semantic impact |
| Missing function word ("government of" → "government") | hallucination (phonetic) | Acoustic event dropped |
| Extra filler word ("um/uh/er" added) | faithful_dialect if in audio; hallucination (noise) if not | Check audio |
| Proper name misspelling | hallucination (phonetic) | Named entities should be correct |
| Numbers ("twenty" → "20") | faithful_dialect | Orthographic variant |

---

## Verification

After annotation, run `compute_benchmark_metrics.py` to see:
- How much WER drops when faithful_dialect errors are excluded (`faithful_wer`)
- Which dialects have the highest `dialect_penalty`
- Which models over-normalize the most (`normalization_overcorrection_rate`)