# From the Newsroom to the Comments Section

**Evaluating LLM Alignment with Human Judgments Across Formal and Informal Web Domains**

Vlad-Tudor Parau, Berat Aras, and Alex Despan — Delft University of Technology, 2026

---

## Overview

This repository contains the full data pipeline for a research paper investigating whether large language models maintain consistent alignment with human sentiment judgments across two contrasting web domains: formal editorial news (NewsMTSC) and informal e-commerce reviews (Amazon Reviews Multi EN).

Three models are evaluated as classifiers — Llama 3.2:3B, Gemma3:4B, and Phi3:mini — each run in both zero-shot and domain-specific prompting modes. Feature detection for RQ2 is performed independently by Llama 3.2:3B and Qwen2.5:3B, enabling inter-model agreement measurement as a robustness check.

The pipeline fetches human-annotated data, balances it, runs all classifiers, then computes alignment metrics and performs linguistic error analysis — all reproducibly from a single command.

---

## Research Questions

**RQ1** — How consistently do LLMs reproduce human-annotated sentiment across formal vs. informal web domains?

**RQ2** — Which linguistic features contribute most to disagreement between human and LLM judgments?

**RQ3** — Does domain-specific prompting improve alignment relative to a generic zero-shot baseline?

---

## Key Findings

### RQ1 — Cross-domain alignment

| Model | Mode | Domain | Accuracy | F1 (weighted) | Cohen's κ |
|---|---|---|---|---|---|
| Llama 3.2:3B | Zero-shot | NYT (formal) | 0.622 | 0.598 | 0.433 |
| Llama 3.2:3B | Zero-shot | Amazon (informal) | 0.655 | 0.643 | 0.482 |
| Llama 3.2:3B | Domain-specific | NYT (formal) | 0.496 | 0.473 | 0.245 |
| Llama 3.2:3B | Domain-specific | Amazon (informal) | 0.532 | 0.521 | 0.297 |
| Gemma3:4B | Zero-shot | NYT (formal) | 0.651 | 0.637 | 0.476 |
| Gemma3:4B | Zero-shot | Amazon (informal) | 0.610 | 0.548 | 0.416 |
| Gemma3:4B | Domain-specific | NYT (formal) | 0.648 | 0.641 | 0.472 |
| Gemma3:4B | Domain-specific | Amazon (informal) | 0.619 | 0.578 | 0.430 |
| **Phi3:mini** | **Zero-shot** | **NYT (formal)** | **0.695** | **0.689** | **0.542** |
| Phi3:mini | Zero-shot | Amazon (informal) | 0.656 | 0.632 | 0.484 |
| Phi3:mini | Domain-specific | NYT (formal) | 0.641 | 0.642 | 0.463 |
| Phi3:mini | Domain-specific | Amazon (informal) | 0.638 | 0.605 | 0.457 |

Phi3:mini achieves the strongest alignment overall (zero-shot NYT: κ=0.542), while Gemma3:4B performs most consistently across both domains. All models show a small advantage on informal reviews in zero-shot mode, though the gap is narrower than in binary-classification settings due to the additional neutral class.

### RQ3 — Effect of domain-specific prompting

| Model | Domain | Accuracy Δ | Kappa Δ | F1 Δ |
|---|---|---|---|---|
| Llama 3.2:3B | NYT | −12.6pp | −0.187 | −0.126 |
| Llama 3.2:3B | Amazon | −12.2pp | −0.185 | −0.122 |
| Gemma3:4B | NYT | −0.3pp | −0.004 | +0.004 |
| Gemma3:4B | Amazon | +0.9pp | +0.014 | +0.030 |
| Phi3:mini | NYT | −5.4pp | −0.080 | −0.048 |
| Phi3:mini | Amazon | −1.8pp | −0.027 | −0.027 |

Models respond very differently to domain-specific prompting. Llama suffers large degradation in both domains (McNemar p≈0), suggesting it over-interprets domain cues and forces neutral or hedged text into sentiment categories. Gemma3 is effectively immune to the prompting change (Δκ < 0.02 in both domains). Phi3 falls between the two with modest but significant degradation on NYT.

### RQ2 — Linguistic feature ranking (LLM-detected)

Features were detected independently by Llama 3.2:3B and Qwen2.5:3B, then ranked by disagreement uplift (P(disagree | feature present) − P(disagree | feature absent)), averaged across all 12 runs (3 models × 2 modes × 2 domains):

| Feature | Mean uplift |
|---|---|
| Hedging | **+0.097** |
| Slang | −0.017 |
| Implicit meaning | −0.022 |
| Sarcasm | −0.047 |
| Mixed sentiment | −0.103 |

Hedging is the only consistent positive predictor of human–LLM disagreement, and the effect is robust across all three classifiers. Mixed sentiment shows the strongest negative uplift: when a model detects both positive and negative signals in the same text, it tends to agree with the human label, likely because the net polarity is still legible. Sarcasm also shows negative uplift — consciously detected irony does not mislead the models.

### RQ2 — Inter-model feature agreement (Llama 3.2:3B vs Qwen2.5:3B)

To assess robustness of the feature labels used in RQ2, the same five features were detected independently by both models. Agreement is reported as raw rate and Cohen's κ, averaged across domains:

| Feature | Agreement | Cohen's κ |
|---|---|---|
| Sarcasm | 0.877 | **0.430** |
| Hedging | 0.658 | 0.301 |
| Slang | 0.943 | 0.195 |
| Mixed sentiment | 0.669 | 0.077 |
| Implicit meaning | 0.557 | 0.033 |

Sarcasm (κ=0.43) and hedging (κ=0.30) show moderate inter-model agreement, lending confidence to the findings for those features. Mixed sentiment and implicit meaning fall near chance-level κ, meaning the two annotator models label those features inconsistently. Findings for these features should be treated with caution.

---

## Dataset Details

| Dataset | Domain | Source | Labels | Samples used |
|---|---|---|---|---|
| [NewsMTSC](https://github.com/fhamborg/NewsMTSC) | Formal — news sentences | fhamborg/NewsMTSC (GitHub) | negative / neutral / positive (expert-annotated) | 498 (balanced, 166/class) |
| [Amazon Reviews Multi EN](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) | Informal — product reviews | SetFit/amazon_reviews_multi_en (HuggingFace) | negative / neutral / positive (star ratings: 1-2 / 3 / 4-5) | 498 (balanced, 166/class) |

---

## Pipeline

```
fetch  →  clean  →  classify  →  evaluate  →  compare
```

### Stage descriptions

| Stage | Command | What it does |
|---|---|---|
| `fetch` | `--stage fetch` | Downloads NewsMTSC JSONL splits from GitHub; loads Amazon Reviews Multi EN from HuggingFace |
| `clean` | `--stage clean` | Deduplicates, normalises labels, downsamples to balanced class distribution |
| `classify` | `--stage classify` | Runs Llama 3.2:3B, Gemma3:4B, and Phi3:mini in zero-shot and domain-specific modes; writes one prediction CSV per (model, mode, domain) |
| `evaluate` | `--stage evaluate` | Computes accuracy/F1/κ per run; runs LLM-based linguistic feature detection (Llama + Qwen); produces error CSVs, feature summaries, and inter-model agreement JSONs |
| `compare` | `--stage compare` | Cross-domain table, cross-model pivot tables, prompt delta table, McNemar tests, aggregated feature ranking, inter-model feature agreement |

### Models

| Role | Models |
|---|---|
| Classification (RQ1, RQ3) | Llama 3.2:3B, Gemma3:4B, Phi3:mini |
| Feature detection (RQ2) | Llama 3.2:3B, Qwen2.5:3B (independent annotators) |

### Prompting modes

- **Zero-shot**: `"You are a sentiment classifier. Respond with exactly one word: positive, neutral, or negative."`
- **Domain-specific (NYT)**: Instructs the model to treat the text as formal editorial writing containing hedging, nuanced argumentation, and subtle critique.
- **Domain-specific (Amazon)**: Instructs the model to treat the text as informal product reviews containing slang, sarcasm, mixed opinions, and colloquial expressions.

### Linguistic feature detection

Features are detected by querying **both Llama 3.2:3B and Qwen2.5:3B** independently with a structured JSON prompt asking each to classify each text for five binary features. Results are cached to `data/processed/llm_features_cache.jsonl` (keyed by text MD5, per model) so detection only runs once per unique text. Per-run inter-model agreement (raw rate + Cohen's κ) is written to `{domain}_{model_slug}_model_agreement.json` and aggregated in `model_agreement_combined.csv`. VADER compound score is retained as a supplementary numeric column.

---

## Repository Structure

```
.
├── config.yaml                    # Model, dataset, and output configuration
├── requirements.txt
├── src/
│   ├── pipeline.py                # Top-level orchestration
│   ├── fetch/
│   │   ├── fetch_nyt.py           # Downloads NewsMTSC from GitHub raw URLs
│   │   └── fetch_amazon.py        # Loads Amazon Reviews Multi EN from HuggingFace
│   ├── clean/
│   │   ├── clean_nyt.py           # Label normalisation + class balancing
│   │   └── clean_amazon.py        # Label mapping (star ratings → 3 classes) + class balancing
│   ├── classify/
│   │   ├── classifier.py          # Ollama client + response parser (model-agnostic)
│   │   └── prompts.py             # Zero-shot and domain-specific prompt templates
│   └── evaluate/
│       ├── metrics.py             # Accuracy, F1, Cohen's κ
│       ├── error_analysis.py      # Per-sample feature flags + inter-model agreement JSON
│       ├── llm_features.py        # Dual-model feature detection with caching (Llama + Qwen)
│       └── compare.py             # Cross-domain/cross-model synthesis + McNemar tests
├── data/
│   ├── raw/                       # Downloaded JSONL files
│   └── processed/                 # Balanced CSVs + feature cache
└── results/
    ├── predictions/               # LLM output CSVs (zero_shot/, domain_specific/)
    ├── metrics/                   # Per-run metrics + comparison tables
    └── error_analysis/            # Per-sample error CSVs + feature summaries
```

---

## Setup and Reproduction

### Requirements

- Python 3.10+
- Docker with the `ollama/ollama` image and `llama3.2:3b`, `gemma3:4b`, `phi3:mini`, and `qwen2.5:3b` models pulled
- A HuggingFace token in `.env` as `HF_TOKEN=...`

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start the models

```bash
docker start ollama   # assumes container already created with: docker run -d -p 11434:11434 --name ollama ollama/ollama
docker exec ollama ollama pull llama3.2:3b
docker exec ollama ollama pull gemma3:4b
docker exec ollama ollama pull phi3:mini
docker exec ollama ollama pull qwen2.5:3b
```

### Run the full pipeline

```bash
python -m src.pipeline              # all stages
python -m src.pipeline --stage fetch
python -m src.pipeline --stage clean
python -m src.pipeline --stage classify
python -m src.pipeline --stage evaluate
python -m src.pipeline --stage compare
```

---

## Output Files

| File | Description |
|---|---|
| `results/predictions/{mode}/{domain}_{model_slug}.csv` | Raw LLM predictions per (model, mode, domain) |
| `results/metrics/{mode}/{domain}_{model_slug}_metrics.json` | Accuracy, F1, κ, classification report |
| `results/metrics/cross_domain_table.csv` | All 12 runs (3 models × 2 modes × 2 domains) side by side |
| `results/metrics/cross_model_{mode}.csv` | Models compared head-to-head per mode |
| `results/metrics/prompt_delta_table.csv` | Domain-specific minus zero-shot deltas per model × domain |
| `results/metrics/mcnemar_tests.json` | Statistical significance of prompting effect per model × domain |
| `results/error_analysis/{mode}/{domain}_{model_slug}_errors.csv` | Per-sample predictions + feature flags (Llama and Qwen columns) |
| `results/error_analysis/{mode}/{domain}_{model_slug}_feature_summary.json` | Feature disagreement rates ranked by uplift |
| `results/error_analysis/{mode}/{domain}_{model_slug}_model_agreement.json` | Per-feature Llama vs Qwen agreement rate and Cohen's κ |
| `results/error_analysis/feature_ranking_combined.csv` | Aggregated feature uplift ranking across all runs and models |
| `results/error_analysis/model_agreement_combined.csv` | Inter-model feature agreement aggregated across all runs |
| `data/processed/llm_features_cache.jsonl` | Cached LLM feature labels (keyed by MD5, per model) |
