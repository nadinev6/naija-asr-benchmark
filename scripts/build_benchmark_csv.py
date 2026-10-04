import json, csv

# Load
with open("data/kaggle_benchmark/analysis/results.json") as f:
    results = json.load(f)

summary = results["summary"]["models"]
metrics = results["metrics"]

DISPLAY = {
    "whisper-large-v3": "Whisper large-v3",
    "whisper-small": "Whisper small",
    "naija-finetuned": "Accent Labs fine-tuned",
}
DIALECTS = ["hausa", "igbo", "multilingual", "pidgin", "yoruba"]

rows = []

# --- summary ---
for mid, m in summary.items():
    d = DISPLAY[mid]
    # Overall overcorrection rate = sum of dialect_normalization across all dialects / total hallucinations
    total_norm = sum(metrics[f"{mid}::{d2}"]["cause_distribution"]["dialect_normalization"] for d2 in DIALECTS)
    total_hall = m["hallucinations"]
    ovcorr = round(total_norm / total_hall * 100, 1) if total_hall else 0
    rows.append(["summary", d, "overall", "raw_wer", f"{m['raw_wer']*100:.2f}"])
    rows.append(["summary", d, "overall", "faithful_wer", f"{m['faithful_wer']*100:.2f}"])
    rows.append(["summary", d, "overall", "dialect_penalty", f"{m['dialect_penalty']*100:.2f}"])
    rows.append(["summary", d, "overall", "hallucination_rate", f"{m['hallucination_rate']*100:.1f}"])
    rows.append(["summary", d, "overall", "overcorrection_pct", f"{ovcorr}"])

# --- per_dialect ---
for mid, display in DISPLAY.items():
    for d in DIALECTS:
        m = metrics.get(f"{mid}::{d}", {})
        rows.append(["per_dialect", display, d, "raw_wer", f"{m.get('raw_wer', 0)*100:.2f}"])
        rows.append(["per_dialect", display, d, "dialect_penalty", f"{m.get('dialect_penalty', 0)*100:.4f}"])
        cause = m.get("cause_distribution", {})
        rows.append(["per_dialect", display, d, "total_words", str(m.get("total_words", 0))])
        rows.append(["per_dialect", display, d, "total_errors", str(m.get("total_errors", 0))])
        rows.append(["per_dialect", display, d, "hallucinations", str(m.get("hallucinations", 0))])
        rows.append(["per_dialect", display, d, "faithful_variants", str(m.get("faithful_dialect_variants", 0))])
        rows.append(["per_dialect", display, d, "overcorrection_rate", f"{m.get('normalization_overcorrection_rate', 0)*100:.1f}"])

# --- cause_distribution (overall per model, % of hallucinations) ---
for mid, display in DISPLAY.items():
    total_phon = sum(metrics[f"{mid}::{d}"]["cause_distribution"]["phonetic"] for d in DIALECTS)
    total_norm = sum(metrics[f"{mid}::{d}"]["cause_distribution"]["dialect_normalization"] for d in DIALECTS)
    total_noise = sum(metrics[f"{mid}::{d}"]["cause_distribution"]["noise"] for d in DIALECTS)
    total_hall = summary[mid]["hallucinations"]
    rows.append(["cause_distribution", display, "overall", "phonetic_pct", f"{total_phon/total_hall*100:.1f}"])
    rows.append(["cause_distribution", display, "overall", "dialect_norm_pct", f"{total_norm/total_hall*100:.1f}"])
    rows.append(["cause_distribution", display, "overall", "noise_pct", f"{total_noise/total_hall*100:.1f}"])
    rows.append(["cause_distribution", display, "overall", "phonetic_count", str(total_phon)])
    rows.append(["cause_distribution", display, "overall", "dialect_norm_count", str(total_norm)])
    rows.append(["cause_distribution", display, "overall", "noise_count", str(total_noise)])

# --- heldout ---
heldout = {"Whisper large-v3": "39.50", "Whisper small": "44.52", "Accent Labs fine-tuned": "7.11"}
for display, wer in heldout.items():
    rows.append(["heldout", display, "overall", "raw_wer", wer])

# Write
path = "data/kaggle_benchmark/analysis/benchmark_eval_results.csv"
with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["section", "model_id", "accent_background", "metric", "value"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {path}")