"""
Main pipeline orchestration.

Usage:
  python -m src.pipeline                        # run all stages
  python -m src.pipeline --stage fetch
  python -m src.pipeline --stage clean
  python -m src.pipeline --stage classify
  python -m src.pipeline --stage evaluate
  python -m src.pipeline --stage compare
"""
import argparse
import pathlib

import yaml
from dotenv import load_dotenv

load_dotenv()  # loads HF_TOKEN (and any other vars) from .env into os.environ

from src.fetch.fetch_nyt import fetch_nyt
from src.fetch.fetch_amazon import fetch_amazon
from src.clean.clean_nyt import clean_nyt
from src.clean.clean_amazon import clean_amazon
from src.classify.classifier import classify
from src.evaluate.metrics import compute_metrics
from src.evaluate.error_analysis import analyze_errors
from src.evaluate.compare import compare


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_fetch(cfg: dict) -> None:
    fetch_nyt(cfg)
    fetch_amazon(cfg)


def stage_clean(cfg: dict) -> None:
    clean_nyt(cfg)
    clean_amazon(cfg)


def _model_slug(model: str) -> str:
    return model.replace(":", "-").replace(".", "")


def stage_classify(cfg: dict) -> None:
    processed = pathlib.Path(cfg["output"]["processed_dir"])
    results = pathlib.Path(cfg["output"]["results_dir"])
    classify_models = cfg["ollama"].get("classify_models", [cfg["ollama"]["model"]])

    for model in classify_models:
        slug = _model_slug(model)
        for domain in ("nyt", "amazon"):
            input_csv = processed / domain / "labeled.csv"
            for mode in ("zero_shot", "domain_specific"):
                output_csv = results / "predictions" / mode / f"{domain}_{slug}.csv"
                classify(input_csv, output_csv, domain=domain, mode=mode, cfg=cfg, model=model)


def stage_evaluate(cfg: dict) -> None:
    results = pathlib.Path(cfg["output"]["results_dir"])
    metrics_dir = results / "metrics"
    errors_dir = results / "error_analysis"
    classify_models = cfg["ollama"].get("classify_models", [cfg["ollama"]["model"]])

    for model in classify_models:
        slug = _model_slug(model)
        for mode in ("zero_shot", "domain_specific"):
            for domain in ("nyt", "amazon"):
                pred_csv = results / "predictions" / mode / f"{domain}_{slug}.csv"
                if not pred_csv.exists():
                    print(f"[pipeline] skipping missing {pred_csv}")
                    continue
                compute_metrics(pred_csv, metrics_dir / mode)
                analyze_errors(pred_csv, errors_dir / mode, cfg=cfg)


def stage_compare(cfg: dict) -> None:
    compare(pathlib.Path(cfg["output"]["results_dir"]), cfg=cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["fetch", "clean", "classify", "evaluate", "compare", "all"],
        default="all",
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stages = {
        "fetch": stage_fetch,
        "clean": stage_clean,
        "classify": stage_classify,
        "evaluate": stage_evaluate,
        "compare": stage_compare,
    }

    if args.stage == "all":
        for fn in stages.values():
            fn(cfg)
    else:
        stages[args.stage](cfg)


if __name__ == "__main__":
    main()
