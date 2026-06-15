"""
LLM-based linguistic feature detection using the same Ollama model as classification.

For each text, asks Llama to return a JSON object with five boolean features:
  sarcasm, slang, hedging, mixed_sentiment, implicit_meaning

Results are cached to data/processed/llm_features_cache.jsonl (keyed by MD5 hash
of the text) so the Ollama calls only happen once across all evaluate runs.
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

"sarcasm"          — true if the text uses irony or means the opposite of what is said
"slang"            — true if the text contains internet slang, abbreviations, or very informal expressions
"hedging"          — true if the text uses epistemic hedges (might, perhaps, allegedly, reportedly, appears to, seems)
"mixed_sentiment"  — true if the text contains both positive and negative sentiment
"implicit_meaning" — true if the sentiment is implied by events or facts rather than stated with explicit sentiment words

Text: {text}"""


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _parse_response(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("no JSON object found")
    parsed = json.loads(raw[start:end])
    return {k: bool(parsed.get(k, False)) for k in FEATURES}


def detect_features(
    texts: list[str],
    cfg: dict,
    cache_path: pathlib.Path = _CACHE_PATH,
) -> pd.DataFrame:
    cache_path = pathlib.Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cache[entry["hash"]] = {k: entry[k] for k in FEATURES}

    client = ollama.Client(host=cfg["ollama"]["base_url"])
    model = cfg["ollama"]["model"]

    to_classify = [(t, _md5(t)) for t in texts if _md5(t) not in cache]
    n_cached = len(texts) - len(to_classify)
    if n_cached:
        print(f"[llm_features] {n_cached} texts loaded from cache, {len(to_classify)} new")

    if to_classify:
        with cache_path.open("a") as cache_f:
            for text, h in tqdm(to_classify, desc="LLM feature detection"):
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

                cache[h] = feats
                cache_f.write(
                    json.dumps({"hash": h, **feats}) + "\n"
                )

    rows = []
    for text in texts:
        row = {"text": text}
        row.update(cache.get(_md5(text), {k: False for k in FEATURES}))
        rows.append(row)

    return pd.DataFrame(rows)
