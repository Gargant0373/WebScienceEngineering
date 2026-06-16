"""
Fetch SetFit/amazon_reviews_multi_en dataset (English product reviews with 0-4 star labels)
and save raw records to data/raw/amazon/raw.jsonl.

Labels: 0=1-star, 1=2-star, 2=3-star, 3=4-star, 4=5-star
Mapped in clean step: 0,1→negative  2→neutral  3,4→positive
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
