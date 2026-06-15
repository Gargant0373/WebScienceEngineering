"""
Run Llama 3 via Ollama over a processed CSV and write predictions.

Input CSV:  text, human_label
Output CSV: text, human_label, llm_label
"""
import pathlib
import re

import ollama
import pandas as pd
from tqdm import tqdm

from src.classify.prompts import build_prompt

_VALID_LABELS = {"positive", "neutral", "negative"}


def _parse_response(raw: str) -> str:
    token = raw.strip().lower().split()[0] if raw.strip() else ""
    token = re.sub(r"[^a-z]", "", token)
    return token if token in _VALID_LABELS else "unknown"


def classify(
    input_csv: pathlib.Path,
    output_csv: pathlib.Path,
    domain: str,
    mode: str,
    cfg: dict,
) -> pathlib.Path:
    df = pd.read_csv(input_csv)
    model = cfg["ollama"]["model"]
    base_url = cfg["ollama"]["base_url"]

    client = ollama.Client(host=base_url)

    labels = []
    for text in tqdm(df["text"], desc=f"Classifying {domain} [{mode}]"):
        messages = build_prompt(str(text), mode=mode, domain=domain)
        response = client.chat(model=model, messages=messages)
        raw = response["message"]["content"]
        labels.append(_parse_response(raw))

    df["llm_label"] = labels
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[classify] {domain}/{mode} → {output_csv}")
    return output_csv
