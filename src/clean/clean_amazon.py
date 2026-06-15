"""
Clean amazon_polarity raw records into a balanced labeled CSV.

Input columns:
  content   - review text
  label     - 0=negative, 1=positive

Output CSV columns:
  text, human_label   (values: positive / negative)
"""
import json
import pathlib
import re

import pandas as pd


_LABEL_MAP = {0: "negative", 1: "positive"}


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_amazon(cfg: dict) -> pathlib.Path:
    raw_path = pathlib.Path(cfg["output"]["raw_dir"]) / "amazon" / "raw.jsonl"
    out_dir = pathlib.Path(cfg["output"]["processed_dir"]) / "amazon"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "labeled.csv"

    rows = []
    with raw_path.open() as f:
        for line in f:
            record = json.loads(line)
            label = _LABEL_MAP.get(record.get("label"))
            if label is None:
                continue
            text = _clean_text(record.get("content", ""))
            if len(text) < 10:
                continue
            rows.append({"text": text, "human_label": label})

    df = pd.DataFrame(rows).drop_duplicates(subset="text")

    sample_size = cfg["datasets"]["amazon"]["sample_size"]
    df = _balance(df, sample_size, cfg.get("seed", 42))

    df.to_csv(out_path, index=False)
    print(f"[clean_amazon] {len(df)} balanced samples → {out_path}")
    return out_path


def _balance(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    per_class = total // df["human_label"].nunique()
    parts = [
        g.sample(n=min(per_class, len(g)), random_state=seed)
        for _, g in df.groupby("human_label")
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
