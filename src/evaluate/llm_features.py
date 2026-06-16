"""
LLM-based linguistic feature detection using multiple Ollama models.

Models:
- llama3.2:3b
- qwen2.5:3b

Each text is annotated independently by both models for:
  sarcasm, slang, hedging, mixed_sentiment, implicit_meaning

Cache format (JSONL):
{
  "hash": "...",
  "llama3.2:3b": {...features...},
  "qwen2.5:3b": {...features...}
}
"""

import hashlib
import json
import pathlib

import ollama
import pandas as pd
from tqdm import tqdm

FEATURES = ["sarcasm", "slang", "hedging", "mixed_sentiment", "implicit_meaning"]

_CACHE_PATH = pathlib.Path("data/processed/llm_features_cache.jsonl")

_SYSTEM = (
    "You are a precise linguistic classifier. "
    "Respond with only a valid JSON object and no other text."
)

_USER_TEMPLATE = """\
Classify the following text for five linguistic features.
Return a JSON object with exactly these boolean keys:

"sarcasm"
"slang"
"hedging"
"mixed_sentiment"
"implicit_meaning"

Definitions:
- sarcasm: irony or opposite meaning
- slang: informal expressions, abbreviations, internet slang
- hedging: uncertainty (might, seems, appears, reportedly)
- mixed_sentiment: both positive and negative sentiment
- implicit_meaning: sentiment implied rather than explicit

Text: {text}
"""


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _parse_response(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("no JSON found")

    parsed = json.loads(raw[start:end])
    return {k: bool(parsed.get(k, False)) for k in FEATURES}


def detect_features(texts: list[str], cfg: dict) -> pd.DataFrame:
    cache_path = pathlib.Path(_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    models = cfg["ollama"].get(
        "models",
        [cfg["ollama"]["model"]]
    )

    # -----------------------
    # Load cache
    # -----------------------
    cache = {}
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    cache[entry["hash"]] = entry

    client = ollama.Client(host=cfg["ollama"]["base_url"])

    to_process = []
    for t in texts:
        h = _md5(t)
        if h not in cache:
            cache[h] = {"hash": h}
        to_process.append((t, h))

    # -----------------------
    # Run missing annotations
    # -----------------------
    with cache_path.open("a") as cache_f:
        for text, h in tqdm(to_process, desc="LLM feature detection (multi-model)"):

            changed = False
            for model in models:

                # skip if already computed
                if model in cache[h]:
                    continue

                prompt = _USER_TEMPLATE.format(text=text[:600])

                try:
                    response = client.chat(
                        model=model,
                        messages=[
                            {"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        options={"temperature": 0},
                    )

                    feats = _parse_response(response.message.content.strip())

                except Exception:
                    feats = {k: False for k in FEATURES}

                cache[h][model] = feats
                changed = True

            if changed:
                cache_f.write(json.dumps(cache[h]) + "\n")

    # -----------------------
    # Build dataframe (default = llama)
    # -----------------------
    rows = []
    for text in texts:
        h = _md5(text)
        entry = cache.get(h, {})

        row = {"text": text}

        # keep llama as default features for existing pipeline
        llama_feats = entry.get("llama3.2:3b", {k: False for k in FEATURES})
        row.update(llama_feats)

        # ALSO store qwen (for RQ2 upgrade later)
        qwen_feats = entry.get("qwen2.5:3b", {k: False for k in FEATURES})
        for k in FEATURES:
            row[f"qwen_{k}"] = qwen_feats.get(k, False)

        rows.append(row)

    return pd.DataFrame(rows)