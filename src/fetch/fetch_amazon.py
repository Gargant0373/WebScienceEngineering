"""
Fetch amazon_polarity dataset (product reviews with binary sentiment labels)
and save raw records to data/raw/amazon/raw.jsonl.

amazon_polarity labels: 0=negative (1-2 stars), 1=positive (4-5 stars)
"""
import json
import pathlib

from datasets import load_dataset
from tqdm import tqdm


def fetch_amazon(cfg: dict) -> pathlib.Path:
    out_dir = pathlib.Path(cfg["output"]["raw_dir"]) / "amazon"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw.jsonl"

    ds_cfg = cfg["datasets"]["amazon"]
    ds = load_dataset(ds_cfg["hf_name"], split=ds_cfg["split"])

    with out_path.open("w") as f:
        for row in tqdm(ds, desc="Fetching Amazon"):
            f.write(json.dumps(row) + "\n")

    print(f"[fetch_amazon] saved {len(ds)} records → {out_path}")
    return out_path
