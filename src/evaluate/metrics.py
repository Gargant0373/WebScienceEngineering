"""
Compute accuracy, precision, recall, F1, and Cohen's Kappa
for a predictions CSV, then save to results/metrics/.
"""
import json
import pathlib

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
)


def compute_metrics(predictions_csv: pathlib.Path, out_dir: pathlib.Path) -> dict:
    df = pd.read_csv(predictions_csv)
    df = df[df["llm_label"] != "unknown"]  # drop unparseable responses

    y_true = df["human_label"].tolist()
    y_pred = df["llm_label"].tolist()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "cohen_kappa": kappa,
        "classification_report": report,
        "n_samples": len(df),
        "n_unknown": int((df["llm_label"] == "unknown").sum()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = predictions_csv.stem
    out_path = out_dir / f"{stem}_metrics.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(
        f"[metrics] accuracy={result['accuracy']:.3f}  "
        f"kappa={kappa:.3f}  → {out_path}"
    )
    return result
