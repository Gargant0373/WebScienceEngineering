"""
Fetch NewsMTSC (real-world subset) from the authoritative GitHub raw URLs
referenced by the original HuggingFace dataset script.

Schema per JSONL line:
  id        - unique sentence id
  sentence  - news sentence text
  mention   - target entity mention
  polarity  - int: -1=negative, 0=neutral, 1=positive
  from, to  - character offsets of the mention
"""
import json
import pathlib

import requests
from tqdm import tqdm


def fetch_nyt(cfg: dict) -> pathlib.Path:
    out_dir = pathlib.Path(cfg["output"]["raw_dir"]) / "nyt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw.jsonl"

    base_url = cfg["datasets"]["nyt"]["base_url"]
    split_files = cfg["datasets"]["nyt"]["splits"]

    rows_written = 0
    with out_path.open("w") as out_f:
        for fname in tqdm(split_files, desc="Fetching NewsMTSC splits"):
            url = base_url + fname
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for line in resp.text.splitlines():
                line = line.strip()
                if line:
                    out_f.write(line + "\n")
                    rows_written += 1

    print(f"[fetch_nyt] saved {rows_written} records → {out_path}")
    return out_path
