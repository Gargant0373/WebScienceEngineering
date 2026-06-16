"""
Cross-domain and cross-prompt comparison of evaluation results.

Reads the 4 metrics JSONs and 4 feature-summary JSONs produced by the
evaluate stage and outputs:
  results/metrics/cross_domain_table.csv   - NYT vs Amazon per mode
  results/metrics/prompt_delta_table.csv   - domain_specific vs zero_shot delta per domain
  results/metrics/mcnemar_tests.json       - significance of domain + prompt effects
  results/error_analysis/feature_ranking_combined.csv - features ranked across all runs
"""
import json
import pathlib

import pandas as pd
from scipy.stats import chi2_contingency


def _load_metrics(results_dir: pathlib.Path, mode: str, domain: str) -> dict:
    p = results_dir / "metrics" / mode / f"{domain}_metrics.json"
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def _load_model_agreement(results_dir: pathlib.Path, mode: str, domain: str) -> dict:
    p = results_dir / "error_analysis" / mode / f"{domain}_model_agreement.json"
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def _load_feature_summary(results_dir: pathlib.Path, mode: str, domain: str) -> dict:
    p = results_dir / "error_analysis" / mode / f"{domain}_feature_summary.json"
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def _load_predictions(results_dir: pathlib.Path, mode: str, domain: str) -> pd.DataFrame:
    p = results_dir / "predictions" / mode / f"{domain}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _mcnemar(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Compare two prediction DataFrames on the same samples via McNemar's test."""
    if df_a.empty or df_b.empty:
        return {}
    merged = df_a[["text", "human_label", "llm_label"]].merge(
        df_b[["text", "human_label", "llm_label"]],
        on=["text", "human_label"],
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return {}
    correct_a = merged["llm_label_a"] == merged["human_label"]
    correct_b = merged["llm_label_b"] == merged["human_label"]
    # Contingency table: [[both correct, a correct b wrong], [a wrong b correct, both wrong]]
    n_cc = int((correct_a & correct_b).sum())
    n_cw = int((correct_a & ~correct_b).sum())
    n_wc = int((~correct_a & correct_b).sum())
    n_ww = int((~correct_a & ~correct_b).sum())
    table = [[n_cc, n_cw], [n_wc, n_ww]]
    # Use chi2 on the off-diagonal (McNemar)
    b, c = n_cw, n_wc
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(merged)}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)
    return {
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "n_pairs": len(merged),
        "n_a_correct_b_wrong": n_cw,
        "n_a_wrong_b_correct": n_wc,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def compare(results_dir: pathlib.Path) -> None:
    results_dir = pathlib.Path(results_dir)
    modes = ["zero_shot", "domain_specific"]
    domains = ["nyt", "amazon"]

    # --- Cross-domain + cross-prompt summary table ---
    rows = []
    for mode in modes:
        for domain in domains:
            m = _load_metrics(results_dir, mode, domain)
            if not m:
                continue
            report = m.get("classification_report", {})
            rows.append({
                "mode": mode,
                "domain": domain,
                "accuracy": round(m.get("accuracy", 0), 4),
                "f1_weighted": round(report.get("weighted avg", {}).get("f1-score", 0), 4),
                "precision_weighted": round(report.get("weighted avg", {}).get("precision", 0), 4),
                "recall_weighted": round(report.get("weighted avg", {}).get("recall", 0), 4),
                "cohen_kappa": round(m.get("cohen_kappa", 0), 4),
                "n_samples": m.get("n_samples", 0),
                "n_unknown": m.get("n_unknown", 0),
            })

    table = pd.DataFrame(rows)
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table_path = metrics_dir / "cross_domain_table.csv"
    table.to_csv(table_path, index=False)
    print(f"[compare] cross-domain table → {table_path}")
    print(table.to_string(index=False))

    # --- Prompt delta table (domain_specific - zero_shot) ---
    delta_rows = []
    for domain in domains:
        zs = table[(table["mode"] == "zero_shot") & (table["domain"] == domain)]
        ds = table[(table["mode"] == "domain_specific") & (table["domain"] == domain)]
        if zs.empty or ds.empty:
            continue
        zs, ds = zs.iloc[0], ds.iloc[0]
        delta_rows.append({
            "domain": domain,
            "accuracy_zero_shot": zs["accuracy"],
            "accuracy_domain_specific": ds["accuracy"],
            "accuracy_delta": round(ds["accuracy"] - zs["accuracy"], 4),
            "kappa_zero_shot": zs["cohen_kappa"],
            "kappa_domain_specific": ds["cohen_kappa"],
            "kappa_delta": round(ds["cohen_kappa"] - zs["cohen_kappa"], 4),
            "f1_zero_shot": zs["f1_weighted"],
            "f1_domain_specific": ds["f1_weighted"],
            "f1_delta": round(ds["f1_weighted"] - zs["f1_weighted"], 4),
        })
    delta_table = pd.DataFrame(delta_rows)
    delta_path = metrics_dir / "prompt_delta_table.csv"
    delta_table.to_csv(delta_path, index=False)
    print(f"\n[compare] prompt delta table → {delta_path}")
    print(delta_table.to_string(index=False))

    # --- McNemar significance tests ---
    tests = {}
    # Domain effect: NYT vs Amazon within each mode
    for mode in modes:
        df_nyt = _load_predictions(results_dir, mode, "nyt")
        df_amz = _load_predictions(results_dir, mode, "amazon")
        # Can't directly compare across domains (different texts); use per-domain correct/wrong rates
        # Instead test: prompting effect within each domain
        pass

    # Prompting effect: zero_shot vs domain_specific within each domain
    for domain in domains:
        df_zs = _load_predictions(results_dir, "zero_shot", domain)
        df_ds = _load_predictions(results_dir, "domain_specific", domain)
        result = _mcnemar(df_zs, df_ds)
        tests[f"prompt_effect_{domain}"] = result
        sig = result.get("significant_at_0.05", "N/A")
        p = result.get("p_value", "N/A")
        print(f"\n[compare] McNemar prompt effect ({domain}): p={p}  significant={sig}")

    tests_path = metrics_dir / "mcnemar_tests.json"
    with tests_path.open("w") as f:
        json.dump(tests, f, indent=2)
    print(f"[compare] McNemar tests → {tests_path}")

    # --- Feature ranking combined across all runs ---
    feat_rows = []
    for mode in modes:
        for domain in domains:
            summary = _load_feature_summary(results_dir, mode, domain)
            if not summary:
                continue
            for feat in summary.get("feature_ranking", []):
                feat_rows.append({
                    "mode": mode,
                    "domain": domain,
                    **feat,
                    "overall_disagree_rate": summary["overall_disagree_rate"],
                })
    feat_df = pd.DataFrame(feat_rows)
    feat_path = results_dir / "error_analysis" / "feature_ranking_combined.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"[compare] feature ranking → {feat_path}")

    # Print aggregated rank by mean uplift across all runs
    if not feat_df.empty:
        agg = (
            feat_df.groupby("feature")["uplift"]
            .agg(["mean", "min", "max"])
            .sort_values("mean", ascending=False)
            .round(4)
        )
        print("\n[compare] feature uplift (mean across all runs):")
        print(agg.to_string())

    # --- Inter-model feature agreement (Llama vs Qwen) ---
    agreement_rows = []
    for mode in modes:
        for domain in domains:
            agreement = _load_model_agreement(results_dir, mode, domain)
            for feat, stats in agreement.items():
                agreement_rows.append({"mode": mode, "domain": domain, "feature": feat, **stats})

    if agreement_rows:
        agreement_df = pd.DataFrame(agreement_rows)
        agreement_path = results_dir / "error_analysis" / "model_agreement_combined.csv"
        agreement_df.to_csv(agreement_path, index=False)
        print(f"\n[compare] inter-model agreement → {agreement_path}")

        agg_agreement = (
            agreement_df.groupby("feature")[["agreement", "cohen_kappa"]]
            .mean()
            .sort_values("cohen_kappa", ascending=False)
            .round(4)
        )
        print("\n[compare] Llama vs Qwen feature agreement (mean across all runs):")
        print(agg_agreement.to_string())
