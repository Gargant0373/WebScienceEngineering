"""
Generate the paper figures directly from the committed result tables.

Outputs (vector PDF, sized for ACM single-column width):
    figures/fig1_rq1_kappa.pdf   RQ1 - zero-shot Cohen's kappa, 3 models x 2 domains
    figures/fig2_rq3_delta.pdf   RQ3 - change in kappa from domain-specific prompting
    figures/fig3_rq2_uplift.pdf  RQ2 - mean disagreement uplift per linguistic feature

All values are read from results/ so the figures cannot drift from the tables.
Run with:  python -m src.plots.make_figures
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures"

# Consistent model order and display names used across all figures.
MODELS = ["llama3.2:3b", "gemma3:4b", "phi3:mini"]
MODEL_LABELS = {"llama3.2:3b": "Llama 3.2:3B", "gemma3:4b": "Gemma3:4B", "phi3:mini": "Phi3:mini"}
DOMAIN_LABELS = {"nyt": "NewsMTSC", "amazon": "Amazon"}
DOMAIN_COLORS = {"nyt": "#3b6fb6", "amazon": "#e07b39"}

FEATURE_ORDER = ["hedging", "slang", "implicit_meaning", "sarcasm", "mixed_sentiment"]
FEATURE_LABELS = {
    "hedging": "Hedging",
    "slang": "Slang",
    "implicit_meaning": "Implicit meaning",
    "sarcasm": "Sarcasm",
    "mixed_sentiment": "Mixed sentiment",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def _grouped_bars(value_col, csv_path, ylabel, out_name, annotate_fmt="{:.2f}"):
    """Grouped bar chart: one group per model, one bar per domain."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    width = 0.38
    x = range(len(MODELS))

    for i, domain in enumerate(["nyt", "amazon"]):
        vals = []
        for model in MODELS:
            row = df[(df["model"] == model) & (df["domain"] == domain)]
            vals.append(float(row[value_col].iloc[0]))
        offsets = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offsets, vals, width, label=DOMAIN_LABELS[domain],
                      color=DOMAIN_COLORS[domain])
        ax.bar_label(bars, labels=[annotate_fmt.format(v) for v in vals],
                     padding=2, fontsize=6)

    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=10)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.20)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def fig1_data_filter():
    # cross_domain_table.csv holds both modes; keep only zero_shot for fig 1.
    src = RESULTS / "metrics" / "cross_domain_table.csv"
    df = pd.read_csv(src)
    return df[df["mode"] == "zero_shot"]


def fig2_rq3_delta():
    _grouped_bars(
        value_col="kappa_delta",
        csv_path=RESULTS / "metrics" / "prompt_delta_table.csv",
        ylabel="$\\Delta\\kappa$ (domain-specific $-$ zero-shot)",
        out_name="fig2_rq3_delta.pdf",
        annotate_fmt="{:+.2f}",
    )


def fig3_rq2_uplift():
    df = pd.read_csv(RESULTS / "error_analysis" / "feature_ranking_combined.csv")
    means = df.groupby("feature")["uplift"].mean()
    vals = [means[f] for f in FEATURE_ORDER]
    labels = [FEATURE_LABELS[f] for f in FEATURE_ORDER]
    colors = ["#2a9d4a" if v > 0 else "#c1432e" for v in vals]

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    y = range(len(FEATURE_ORDER))
    bars = ax.barh(list(y), vals, color=colors)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in vals], padding=3, fontsize=6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean disagreement uplift")
    ax.margins(x=0.22)
    ax.axvline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    out = FIG_DIR / "fig3_rq2_uplift.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    # Fig 1 needs the zero-shot subset; write a filtered temp frame to disk-free path.
    zs = fig1_data_filter()
    tmp = FIG_DIR / "_zero_shot_kappa.csv"
    zs.to_csv(tmp, index=False)
    _grouped_bars(
        value_col="cohen_kappa",
        csv_path=tmp,
        ylabel="Cohen's $\\kappa$ (zero-shot)",
        out_name="fig1_rq1_kappa.pdf",
    )
    tmp.unlink()
    fig2_rq3_delta()
    fig3_rq2_uplift()


if __name__ == "__main__":
    main()
