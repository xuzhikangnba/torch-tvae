"""Compare NEW endpoint metrics with the same archived paper condition.

No metrics are overwritten. Per-view MRI values stay separate. This report is diagnostic, not
an automatic assertion that an entire paper has been reproduced.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--targets", type=Path, default=Path(__file__).with_name("paper_targets.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    targets = json.loads(args.targets.read_text())
    rows = []
    for folder in args.runs:
        cfg = json.loads((folder / "run_config.json").read_text())
        manifest = json.loads((folder / "data_manifest.json").read_text())
        actual = json.loads((folder / "final_metrics.json").read_text())["metrics"]["test"]
        if cfg["functionality_check"]:
            raise ValueError(f"Cannot compare functionality-only run {folder}")
        target = next(x for x in targets["targets"]
                      if x["dataset"] == cfg["dataset"] and abs(x["gamma"] - cfg["gamma"]) < 1e-9)
        for name, expected in target["metrics"].items():
            assignment = "refit" if name.startswith("test_refit_") else "native"
            metric = name.removeprefix("test_refit_").removeprefix("test_stage_")
            value = actual[assignment][metric]
            tolerance = {"accuracy": .03, "ari": .05, "nmi": .05,
                         "silhouette": .05, "davies_bouldin": .15 * abs(expected)}[metric]
            delta = None if value is None or expected is None else value - expected
            rows.append({"run": folder.name, "dataset": cfg["dataset"], "gamma": cfg["gamma"],
                         "assignment": assignment, "metric": metric, "paper_condition": expected,
                         "fresh_final": value, "difference": delta, "tolerance": tolerance,
                         "within_band": delta is not None and abs(delta) <= tolerance,
                         "split": manifest["split"], "pretrain_epochs": cfg["pretrain_epochs"],
                         "main_updates": cfg["main_updates"], "kl_weight": cfg["kl_weight"],
                         "selection": "fresh final endpoint versus archived selected epoch"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"compared_cells": len(rows), "within_declared_band": sum(x["within_band"] for x in rows),
                      "output": str(args.output), "full_paper_reproduction_claimed": False}, indent=2))


if __name__ == "__main__":
    main()
