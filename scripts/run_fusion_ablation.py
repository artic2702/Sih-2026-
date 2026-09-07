"""
Runs the MANDATORY 3-way ablation for the Optical-SAR Fusion model:
optical-only vs. SAR-only vs. optical+SAR fused — reported HONESTLY (never
forcing fusion to "win"), per fusion_model.py's own docstring and
docs/DECISIONS.md.

Written as a standalone script rather than going through
src/evaluation/evaluate.py because that file's fusion path currently has a
bug: it collects batch["label"] (a list of class-name strings) directly
into gold_labels and never converts it to the multi-hot binary array
multilabel_f1() actually needs — comparing strings against probabilities
would silently break. Flagged for whoever owns evaluate.py; not fixed here
to stay inside src/training/train_fusion.py + src/models/fusion_model.py +
this script, per the team's "only touch your own files" git workflow.

Usage:
    python scripts/run_fusion_ablation.py --config configs/config.yaml --split test
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.models.fusion_model import FusionModel
from src.preprocessing.dataset_loader import get_dataloader
from src.evaluation.metrics import multilabel_f1, fusion_ablation

DATASET_NAME = "bigearthnet_mm"


def _multihot(label_lists, class_names):
    out = np.zeros((len(label_lists), len(class_names)), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        for lbl in labels:
            if lbl in class_names:
                out[i, class_names.index(lbl)] = 1.0
    return out


def collect_ablation_probs(model: FusionModel, loader, limit: int = None) -> dict:
    """Runs all three prediction legs per sample (predict / predict_optical_only
    / predict_sar_only — all already implemented in fusion_model.py) and
    collects class-probability vectors + gold multi-hot labels.
    """
    fusion_probs, optical_probs, sar_probs, golds = [], [], [], []
    count = 0

    for batch in loader:
        n = len(batch["label"])
        for i in range(n):
            optical_i = batch["image_optical"][i].numpy()
            sar_i = batch["image_sar"][i].numpy()

            fusion_res = model.predict(image_optical=optical_i, image_sar=sar_i, query=None)
            optical_res = model.predict_optical_only(image_optical=optical_i, query=None)
            sar_res = model.predict_sar_only(image_sar=sar_i, query=None)

            fusion_probs.append(list(fusion_res["metadata"]["parameters"]["class_probabilities"].values()))
            optical_probs.append(list(optical_res["metadata"]["parameters"]["class_probabilities"].values()))
            sar_probs.append(list(sar_res["metadata"]["parameters"]["class_probabilities"].values()))
            golds.append(batch["label"][i])
            count += 1
            if limit is not None and count >= limit:
                break
        if limit is not None and count >= limit:
            break

    gold_multihot = _multihot(golds, model.labels)
    return {
        "fusion": np.array(fusion_probs),
        "optical": np.array(optical_probs),
        "sar": np.array(sar_probs),
        "gold": gold_multihot,
    }


def print_report(report: dict, n_samples: int):
    print("\n" + "=" * 60)
    print(f"THREE-WAY FUSION ABLATION  (n={n_samples})")
    print("=" * 60)
    for leg in ("optical", "sar", "fusion"):
        m = report[leg]
        print(f"{leg:8s}  macro_f1={m['macro_f1']:.4f}  micro_f1={m['micro_f1']:.4f}")
    print("-" * 60)
    d = report["deltas"]
    print(f"fusion - optical : {d['fusion_minus_optical']:+.4f}")
    print(f"fusion - sar     : {d['fusion_minus_sar']:+.4f}")
    print(f"best leg         : {report['best_leg']}")
    print(f"fusion genuinely helped both legs: {report['fusion_helped']}")
    print("=" * 60 + "\n")


def write_markdown_table(report: dict, out_path: str, n_samples: int):
    lines = [
        "# Optical-SAR Fusion — 3-Way Ablation\n",
        f"n_samples: {n_samples}\n",
        "| Leg | Macro F1 | Micro F1 |",
        "|---|---|---|",
    ]
    for leg in ("optical", "sar", "fusion"):
        m = report[leg]
        lines.append(f"| {leg} | {m['macro_f1']:.4f} | {m['micro_f1']:.4f} |")
    lines.append("")
    lines.append(f"**Best leg:** {report['best_leg']}  ")
    lines.append(f"**Fusion helped both legs:** {report['fusion_helped']}  ")
    lines.append(f"**Δ fusion−optical:** {report['deltas']['fusion_minus_optical']:+.4f}  ")
    lines.append(f"**Δ fusion−sar:** {report['deltas']['fusion_minus_sar']:+.4f}  ")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote markdown table -> {out_path}")


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="limit number of samples for quick run")
    parser.add_argument("--out-json", default="src/evaluation/results/fusion_ablation.json")
    parser.add_argument("--out-md", default="src/evaluation/results/fusion_ablation.md")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = get_dataloader(task="fusion", split=args.split, dataset=DATASET_NAME,
                             config=config, batch_size=config["training"]["batch_size"])

    model = FusionModel(config=config)
    model.load(config["models"]["fusion"]["checkpoint"], device=args.device)
    if not getattr(model, "_checkpoint_loaded", False):
        print(
            "[run_fusion_ablation] WARNING: no trained checkpoint was found — "
            "this ablation will run against a freshly-initialized fusion head. "
            "Run src/training/train_fusion.py first for a meaningful result."
        )

    collected = collect_ablation_probs(model, loader, limit=args.limit)
    optical_metric = multilabel_f1(collected["optical"], collected["gold"])
    sar_metric = multilabel_f1(collected["sar"], collected["gold"])
    fusion_metric = multilabel_f1(collected["fusion"], collected["gold"])

    report = fusion_ablation(optical_metric, sar_metric, fusion_metric)
    n_samples = len(collected["gold"])
    print_report(report, n_samples)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({**report, "n_samples": n_samples, "split": args.split}, f, indent=2)
    print(f"Wrote JSON -> {args.out_json}")

    write_markdown_table(report, args.out_md, n_samples)


if __name__ == "__main__":
    main()
