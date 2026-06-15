"""
Clean NewsMTSC raw records into a balanced labeled CSV.

Input fields (per JSONL line):
  sentence  - news sentence text
  polarity  - int: -1=negative, 0=neutral, 1=positive

Output CSV columns:
  text, human_label   (values: positive / neutral / negative)
"""
import json
import pathlib
import re

import pandas as pd


_LABEL_MAP = {-1: "negative", 0: "neutral", 1: "positive"}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_nyt(cfg: dict) -> pathlib.Path:
    raw_path = pathlib.Path(cfg["output"]["raw_dir"]) / "nyt" / "raw.jsonl"
    out_dir = pathlib.Path(cfg["output"]["processed_dir"]) / "nyt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "labeled.csv"

    rows = []
    seen_ids = set()
    with raw_path.open() as f:
        for line in f:
            record = json.loads(line)
            record_id = record.get("id")
            if record_id and record_id in seen_ids:
                continue
            if record_id:
                seen_ids.add(record_id)

            label = _LABEL_MAP.get(int(record.get("polarity", 999)))
            if label is None:
                continue
            text = _clean_text(record.get("sentence", ""))
            if len(text) < 10:
                continue
            rows.append({"text": text, "human_label": label})

    df = pd.DataFrame(rows).drop_duplicates(subset="text")

    sample_size = cfg["datasets"]["nyt"]["sample_size"]
    df = _balance(df, sample_size, cfg.get("seed", 42))

    df.to_csv(out_path, index=False)
    print(f"[clean_nyt] {len(df)} balanced samples → {out_path}")
    return out_path


def _balance(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    per_class = total // df["human_label"].nunique()
    parts = [
        g.sample(n=min(per_class, len(g)), random_state=seed)
        for _, g in df.groupby("human_label")
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
