"""Run cohort creation, cyclic application, BERT training/application, and tables."""
from __future__ import annotations
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml
from python.common import load_config
from python.cohorts import build_from_config

STAGES = ("cohorts", "cyclic", "bert", "tables")


def invoke(command: list[str], *, dry_run: bool = False):
    print("$", " ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def run(config_path: str, stages: list[str], *, dry_run: bool = False,
        allow_count_mismatch: bool = False):
    cfg = load_config(config_path)
    output = Path(cfg["paths"]["output_dir"]).resolve()
    cohort_dir = output / "cohorts"
    predictions = output / "predictions"
    model_root = output / "models"
    configs = output / "cohort_configs"
    for path in [output, predictions, model_root, configs]:
        path.mkdir(parents=True, exist_ok=True)
    if "cohorts" in stages:
        if dry_run:
            print("Would construct dataset-specific and pooled cohorts")
        else:
            print(build_from_config(cfg, strict=not allow_count_mismatch).to_string(index=False))
    manifest_path = cohort_dir / "cohort_manifest_PRIVATE.csv"
    if not manifest_path.exists():
        if dry_run and stages == ["cohorts"]:
            return
        raise FileNotFoundError(f"Run the cohorts stage first: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    available = {domain: set(manifest.loc[manifest.domain == domain, "cohort"])
                 for domain in ("dataset_1", "dataset_2")}
    if "cyclic" in stages:
        for domain in ("dataset_1", "dataset_2"):
            for slug in sorted(available[domain]):
                source = cohort_dir / f"{domain}_{slug}_PRIVATE.csv"
                destination = predictions / f"{domain}_cyclic_{slug}_predictions_PRIVATE.csv"
                cyclic = cfg["cyclic"]
                invoke(["Rscript", "R/run_cyclic.R", str(source), str(destination),
                        cyclic["patient_id"], cyclic["drug_name"], cyclic["date"],
                        f"{cyclic['i_previous_records']},{cyclic['j_days']}"], dry_run=dry_run)
    if "bert" in stages:
        for slug in sorted(available["dataset_1"]):
            scoped = copy.deepcopy(cfg)
            scoped["paths"]["dataset_1_csv"] = str(cohort_dir / f"dataset_1_{slug}_PRIVATE.csv")
            scoped["paths"]["output_dir"] = str(model_root / slug)
            if slug in available["dataset_2"]:
                scoped["paths"]["dataset_2_csv"] = str(cohort_dir / f"dataset_2_{slug}_PRIVATE.csv")
            scoped_config = configs / f"{slug}.yaml"
            if not dry_run:
                scoped_config.write_text(yaml.safe_dump(scoped, sort_keys=False), encoding="utf-8")
            invoke([sys.executable, "-m", "python.train_bert", "--config", str(scoped_config)], dry_run=dry_run)
            invoke([sys.executable, "-m", "python.apply_bert", "--config", str(scoped_config),
                    "--dataset", "dataset_1_holdout"], dry_run=dry_run)
            source = model_root / slug / "dataset_1_holdout_bert_predictions_PRIVATE.csv"
            dest = predictions / f"dataset_1_bert_{slug}_predictions_PRIVATE.csv"
            if not dry_run:
                dest.write_bytes(source.read_bytes())
            if slug in available["dataset_2"]:
                invoke([sys.executable, "-m", "python.apply_bert", "--config", str(scoped_config),
                        "--dataset", "dataset_2"], dry_run=dry_run)
                source = model_root / slug / "dataset_2_bert_predictions_PRIVATE.csv"
                dest = predictions / f"dataset_2_bert_{slug}_predictions_PRIVATE.csv"
                if not dry_run:
                    dest.write_bytes(source.read_bytes())
    if "tables" in stages:
        invoke([sys.executable, "-m", "tables.generate", "--predictions-dir", str(predictions),
                "--output-dir", str(output / "manuscript_tables"),
                "--dataset_1-cohort", cfg["paths"]["dataset_1_csv"],
                "--dataset_2-cohort", cfg["paths"]["dataset_2_csv"],
                "--bootstrap", str(cfg["analysis"]["bootstrap_iterations"]),
                "--seed", str(cfg["analysis"]["random_seed"])], dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()
    run(args.config, args.stages, dry_run=args.dry_run,
        allow_count_mismatch=args.allow_count_mismatch)

if __name__ == "__main__":
    main()
