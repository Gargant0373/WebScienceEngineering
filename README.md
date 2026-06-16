# From the Newsroom to the Comments Section

**Evaluating LLM Alignment with Human Judgments Across Formal and Informal Web Domains**

Vlad-Tudor Parau, Berat Aras, and Alex Despan — Delft University of Technology, 2026

---

## Overview

This repository contains the full data pipeline for a research paper investigating whether a large language model (Llama 3.2:3B) maintains consistent alignment with human sentiment judgments across two contrasting web domains: formal editorial news (NewsMTSC) and informal e-commerce reviews (Amazon Reviews Multi EN).

The pipeline fetches human-annotated data, balances it, runs the LLM in two prompting modes (zero-shot and domain-specific), then computes alignment metrics and performs linguistic error analysis — all reproducibly from a single command.

---

## Research Questions

**RQ1** — How consistently does the LLM reproduce human-annotated sentiment across formal vs. informal web domains?

**RQ2** — Which linguistic features contribute most to disagreement between human and LLM judgments?

**RQ3** — Does domain-specific prompting improve alignment relative to a generic zero-shot baseline?

---

## Key Findings

### RQ1 — Cross-domain alignment gap

| Mode | Domain | Accuracy | F1 (weighted) | Cohen's κ |
|---|---|---|---|---|
| Zero-shot | NYT (formal) | 0.608 | 0.586 | 0.411 |
| Zero-shot | Amazon (informal) | **0.874** | **0.907** | **0.765** |
| Domain-specific | NYT (formal) | 0.492 | 0.472 | 0.239 |
| Domain-specific | Amazon (informal) | 0.620 | 0.722 | 0.422 |

The LLM aligns substantially better with informal review sentiment than formal editorial language in zero-shot mode (87.4% vs 60.8% accuracy; κ=0.765 vs κ=0.411).

### RQ3 — Domain-specific prompting degraded performance

| Domain | Accuracy Δ | Kappa Δ | F1 Δ |
|---|---|---|---|
| NYT | −11.6pp | −0.172 | −0.114 |
| Amazon | −25.4pp | −0.342 | −0.185 |

Domain-specific prompting *hurt* both domains significantly (McNemar's test, p≈0 for both). This suggests that explicit domain priming causes the model to over-interpret neutral or hedged text as sentiment-bearing.

### RQ2 — Linguistic feature ranking (LLM-detected)

Features were detected by querying the same Llama model for each text, ranked by disagreement uplift (P(disagree | feature present) − P(disagree | feature absent)):

| Feature | Mean uplift |
|---|---|
| Hedging | +0.023 |
| Slang | +0.010 |
| Implicit meaning | −0.008 |
| Mixed sentiment | −0.043 |
| Sarcasm | **−0.183** |

Hedging is the strongest positive predictor of disagreement. Sarcasm shows a strongly negative uplift — texts the LLM identifies as sarcastic are ones it tends to classify *in agreement* with humans, suggesting the model is internally consistent when it consciously detects irony.

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
| `classify` | `--stage classify` | Calls Ollama (Docker) with zero-shot and domain-specific prompts; writes prediction CSVs |
| `evaluate` | `--stage evaluate` | Computes accuracy/F1/κ per run; runs LLM-based linguistic feature detection; produces error CSVs and feature summaries |
| `compare` | `--stage compare` | Cross-domain table, prompt delta table, McNemar significance tests, aggregated feature ranking |

### Prompting modes

- **Zero-shot**: `"You are a sentiment classifier. Respond with exactly one word: positive, neutral, or negative."`
- **Domain-specific (NYT)**: Instructs the model to treat the text as formal editorial writing containing hedging, nuanced argumentation, and subtle critique.
- **Domain-specific (Amazon)**: Instructs the model to treat the text as informal product reviews containing slang, sarcasm, mixed opinions, and colloquial expressions.

### Linguistic feature detection

Features are detected by querying Llama 3.2:3B with a structured JSON prompt asking it to classify each text for five binary features. Results are cached to `data/processed/llm_features_cache.jsonl` so detection only runs once per unique text. VADER compound score is retained as a supplementary numeric column.

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
│   │   └── fetch_amazon.py        # Loads Amazon Polarity from HuggingFace
│   ├── clean/
│   │   ├── clean_nyt.py           # Label normalisation + class balancing
│   │   └── clean_amazon.py        # HTML stripping + class balancing
│   ├── classify/
│   │   ├── classifier.py          # Ollama client + response parser
│   │   └── prompts.py             # Zero-shot and domain-specific prompt templates
│   └── evaluate/
│       ├── metrics.py             # Accuracy, F1, Cohen's κ
│       ├── error_analysis.py      # Per-sample feature flags + summary JSON
│       ├── llm_features.py        # LLM-based linguistic feature detection with caching
│       └── compare.py             # Cross-domain/cross-prompt synthesis + McNemar tests
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
- Docker with the `ollama/ollama` image and `llama3.2:3b` model pulled
- A HuggingFace token (for Amazon Polarity download) in `.env` as `HF_TOKEN=...`

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start the model

```bash
docker start ollama   # assumes container already created with: docker run -d -p 11434:11434 --name ollama ollama/ollama
docker exec ollama ollama pull llama3.2:3b
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
| `results/metrics/{mode}/{domain}_metrics.json` | Accuracy, F1, κ, classification report |
| `results/metrics/cross_domain_table.csv` | All 4 runs side by side |
| `results/metrics/prompt_delta_table.csv` | Domain-specific minus zero-shot deltas |
| `results/metrics/mcnemar_tests.json` | Statistical significance of prompting effect |
| `results/error_analysis/{mode}/{domain}_errors.csv` | Per-sample predictions + feature flags |
| `results/error_analysis/{mode}/{domain}_feature_summary.json` | Feature disagreement rates ranked by uplift |
| `results/error_analysis/feature_ranking_combined.csv` | Aggregated ranking across all runs |
| `data/processed/llm_features_cache.jsonl` | Cached LLM feature labels (keyed by MD5) |
