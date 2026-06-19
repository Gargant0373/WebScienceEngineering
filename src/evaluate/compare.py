"""
Cross-domain, cross-prompt, and cross-model comparison of evaluation results.

Reads the metrics, feature-summary, and model-agreement JSONs produced by the
evaluate stage and outputs:
  results/metrics/cross_domain_table.csv        - all runs (model × mode × domain)
  results/metrics/cross_model_{mode}.csv        - models side by side per mode
  results/metrics/prompt_delta_table.csv        - domain_specific vs zero_shot per (model, domain)
  results/metrics/mcnemar_tests.json            - significance of prompting effect per (model, domain)
  results/error_analysis/feature_ranking_combined.csv
  results/error_analysis/model_agreement_combined.csv
"""
import json
import pathlib

import pandas as pd
from scipy.stats import chi2


def _model_slug(model: str) -> str:
    return model.replace(":", "-").replace(".", "")


def _load_metrics(results_dir: pathlib.Path, mode: str, stem: str) -> dict:
    p = results_dir / "metrics" / mode / f"{stem}_metrics.json"
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def _load_feature_summary(results_dir: pathlib.Path, mode: str, stem: str) -> dict:
    p = results_dir / "error_analysis" / mode / f"{stem}_feature_summary.json"
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def _load_model_agreement(results_dir: pathlib.Path, mode: str, stem: str) -> dict:
    p = results_dir / "error_analysis" / mode / f"{stem}_model_agreement.json"
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def _load_predictions(results_dir: pathlib.Path, mode: str, stem: str) -> pd.DataFrame:
    p = results_dir / "predictions" / mode / f"{stem}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _mcnemar(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
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
    n_cw = int((correct_a & ~correct_b).sum())
    n_wc = int((~correct_a & correct_b).sum())
    b, c = n_cw, n_wc
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(merged)}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(statistic, df=1)
    return {
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "n_pairs": len(merged),
        "n_a_correct_b_wrong": n_cw,
        "n_a_wrong_b_correct": n_wc,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def _holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order.

    Controls the family-wise error rate across the family of McNemar tests.
    Compare each adjusted value against the desired alpha (e.g. 0.05).
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def compare(results_dir: pathlib.Path, cfg: dict = None) -> None:
    results_dir = pathlib.Path(results_dir)
    modes = ["zero_shot", "domain_specific"]
    domains = ["nyt", "amazon"]

    classify_models = []
    if cfg is not None:
        classify_models = cfg["ollama"].get("classify_models", [cfg["ollama"]["model"]])
    if not classify_models:
        raise ValueError("No classify_models found in config")

    slugs = [_model_slug(m) for m in classify_models]
    model_by_slug = {_model_slug(m): m for m in classify_models}

    # --- All-runs table (model × mode × domain) ---
    rows = []
    for slug in slugs:
        for mode in modes:
            for domain in domains:
                stem = f"{domain}_{slug}"
                m = _load_metrics(results_dir, mode, stem)
                if not m:
                    continue
                report = m.get("classification_report", {})
                rows.append({
                    "model": model_by_slug[slug],
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

    # --- Cross-model comparison pivot (one table per mode) ---
    if not table.empty:
        for mode in modes:
            t = table[table["mode"] == mode][
                ["model", "domain", "accuracy", "f1_weighted", "cohen_kappa"]
            ]
            pivot = t.pivot(index="model", columns="domain", values=["accuracy", "f1_weighted", "cohen_kappa"])
            pivot.columns = [f"{metric}_{domain}" for metric, domain in pivot.columns]
            pivot_path = metrics_dir / f"cross_model_{mode}.csv"
            pivot.reset_index().to_csv(pivot_path, index=False)
            print(f"\n[compare] cross-model table ({mode}) → {pivot_path}")
            print(pivot.to_string())

    # --- Prompt delta table (domain_specific − zero_shot per model × domain) ---
    delta_rows = []
    for slug in slugs:
        model = model_by_slug[slug]
        for domain in domains:
            zs = table[(table["model"] == model) & (table["mode"] == "zero_shot") & (table["domain"] == domain)]
            ds = table[(table["model"] == model) & (table["mode"] == "domain_specific") & (table["domain"] == domain)]
            if zs.empty or ds.empty:
                continue
            zs, ds = zs.iloc[0], ds.iloc[0]
            delta_rows.append({
                "model": model,
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

    # --- McNemar tests (prompting effect per model × domain) ---
    tests = {}
    for slug in slugs:
        for domain in domains:
            stem = f"{domain}_{slug}"
            df_zs = _load_predictions(results_dir, "zero_shot", stem)
            df_ds = _load_predictions(results_dir, "domain_specific", stem)
            tests[f"prompt_effect_{domain}_{slug}"] = _mcnemar(df_zs, df_ds)

    # Holm-Bonferroni correction across the family of McNemar tests, so the
    # reported significance accounts for running six paired comparisons.
    valid_keys = [k for k, r in tests.items() if "p_value" in r]
    adjusted = _holm_bonferroni([tests[k]["p_value"] for k in valid_keys])
    for key, p_adj in zip(valid_keys, adjusted):
        tests[key]["p_value_holm"] = round(float(p_adj), 4)
        tests[key]["significant_holm_0.05"] = bool(p_adj < 0.05)

    for slug in slugs:
        model = model_by_slug[slug]
        for domain in domains:
            result = tests[f"prompt_effect_{domain}_{slug}"]
            p = result.get("p_value", "N/A")
            p_adj = result.get("p_value_holm", "N/A")
            sig = result.get("significant_holm_0.05", "N/A")
            print(f"\n[compare] McNemar prompt effect ({domain}, {model}): "
                  f"p={p}  p_holm={p_adj}  significant={sig}")

    tests_path = metrics_dir / "mcnemar_tests.json"
    with tests_path.open("w") as f:
        json.dump(tests, f, indent=2)
    print(f"[compare] McNemar tests → {tests_path}")

    # --- Feature ranking (aggregated across all runs and models) ---
    feat_rows = []
    for slug in slugs:
        for mode in modes:
            for domain in domains:
                stem = f"{domain}_{slug}"
                summary = _load_feature_summary(results_dir, mode, stem)
                if not summary:
                    continue
                for feat in summary.get("feature_ranking", []):
                    feat_rows.append({
                        "model": model_by_slug[slug],
                        "mode": mode,
                        "domain": domain,
                        **feat,
                        "overall_disagree_rate": summary["overall_disagree_rate"],
                    })
    feat_df = pd.DataFrame(feat_rows)
    feat_path = results_dir / "error_analysis" / "feature_ranking_combined.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"[compare] feature ranking → {feat_path}")

    if not feat_df.empty:
        agg = (
            feat_df.groupby("feature")["uplift"]
            .agg(["mean", "min", "max"])
            .sort_values("mean", ascending=False)
            .round(4)
        )
        print("\n[compare] feature uplift (mean across all runs and models):")
        print(agg.to_string())

    # --- Inter-model feature agreement (Llama vs Qwen — independent of classify model) ---
    agreement_rows = []
    seen = set()
    for slug in slugs:
        for mode in modes:
            for domain in domains:
                stem = f"{domain}_{slug}"
                agreement = _load_model_agreement(results_dir, mode, stem)
                for feat, stats in agreement.items():
                    key = (mode, domain, feat)
                    if key in seen:
                        continue
                    seen.add(key)
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
