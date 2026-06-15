"""
Analyze disagreement cases between human labels and LLM predictions.

Feature detection: LLM-based (via llm_features.detect_features).
VADER compound retained as a supplementary numeric column.

Outputs:
  {stem}_errors.csv            - per-sample flags + vader_compound
  {stem}_feature_summary.json  - disagreement rates per feature, ranked by uplift
"""
import json
import pathlib

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.evaluate.llm_features import detect_features, FEATURES

_vader = SentimentIntensityAnalyzer()


def _build_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    n_disagree_total = int(df["disagreement"].sum())

    rows = []
    for feat in FEATURES:
        mask = df[feat]
        n_feat = int(mask.sum())
        n_feat_disagree = int((mask & df["disagreement"]).sum())
        disagree_rate = n_feat_disagree / n_feat if n_feat > 0 else 0.0
        coverage = n_feat_disagree / n_disagree_total if n_disagree_total > 0 else 0.0
        n_no_feat = total - n_feat
        n_no_feat_disagree = int((~mask & df["disagreement"]).sum())
        baseline_rate = n_no_feat_disagree / n_no_feat if n_no_feat > 0 else 0.0
        rows.append({
            "feature": feat,
            "n_present": n_feat,
            "n_disagree_with_feature": n_feat_disagree,
            "disagree_rate": round(disagree_rate, 4),
            "baseline_disagree_rate": round(baseline_rate, 4),
            "uplift": round(disagree_rate - baseline_rate, 4),
            "coverage_of_disagreements": round(coverage, 4),
        })

    rows.sort(key=lambda r: r["uplift"], reverse=True)
    return {
        "total_samples": total,
        "total_disagreements": n_disagree_total,
        "overall_disagree_rate": round(n_disagree_total / total, 4),
        "feature_ranking": rows,
    }


def analyze_errors(
    predictions_csv: pathlib.Path,
    out_dir: pathlib.Path,
    cfg: dict = None,
) -> pathlib.Path:
    df = pd.read_csv(predictions_csv)
    df["disagreement"] = df["human_label"] != df["llm_label"]

    # VADER compound (numeric, supplementary)
    df["vader_compound"] = df["text"].apply(
        lambda t: round(_vader.polarity_scores(str(t))["compound"], 4)
    )

    # LLM-based binary feature flags
    if cfg is not None:
        feat_df = detect_features(df["text"].tolist(), cfg)
        for feat in FEATURES:
            df[feat] = feat_df[feat].values
    else:
        for feat in FEATURES:
            df[feat] = False

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{predictions_csv.stem}_errors.csv"
    df.to_csv(out_path, index=False)

    summary = _build_summary(df)
    summary_path = out_dir / f"{predictions_csv.stem}_feature_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    top = summary["feature_ranking"][0]
    n_disagree = summary["total_disagreements"]
    print(
        f"[error_analysis] {n_disagree}/{len(df)} disagreements | "
        f"top feature: {top['feature']} (uplift={top['uplift']:.3f}) → {out_path}"
    )
    return out_path
