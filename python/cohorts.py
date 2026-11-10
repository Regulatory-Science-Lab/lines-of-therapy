"""Create dataset-specific tumour cohorts and a pooled ('all') cohort."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import pandas as pd
from python.common import load_config


def slugify(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower().replace("&", "and"))).strip("_")


def construct_cohorts(df: pd.DataFrame, *, domain: str, tumour_names: list[str],
                      min_patients: int = 100, expected_tumours: int | None = None,
                      strict: bool = True) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    required = {"patient_id", "tumour_type", "line_number"}
    missing = required - set(df)
    if missing:
        raise ValueError(f"{domain}: missing {sorted(missing)}")
    if df["patient_id"].isna().any() or df["tumour_type"].isna().any():
        raise ValueError(f"{domain}: patient_id and tumour_type must be nonmissing")
    if pd.to_numeric(df["line_number"], errors="coerce").isna().any():
        raise ValueError(f"{domain}: line_number must be numeric and nonmissing")
    if df.empty:
        raise ValueError(f"{domain}: empty source cohort")
    if len(set(tumour_names)) != len(tumour_names):
        raise ValueError("Tumour names must be unique")
    observed = set(df["tumour_type"].astype(str).unique())
    unrecognized = observed - set(tumour_names)
    if unrecognized:
        raise ValueError(f"{domain}: unmapped tumour labels: {sorted(unrecognized)}")
    patient_counts = df.groupby("tumour_type")["patient_id"].nunique()
    eligible = [name for name in tumour_names if patient_counts.get(name, 0) >= min_patients]
    if strict and expected_tumours is not None and len(eligible) != expected_tumours:
        raise ValueError(f"{domain}: expected {expected_tumours} eligible tumour-specific cohorts, "
                         f"found {len(eligible)} ({eligible}). Check source labels and eligibility.")
    frames = {"all": df.copy()}
    for name in eligible:
        slug = slugify(name)
        if slug == "all" or slug in frames:
            raise ValueError(f"Duplicate or reserved tumour slug: {slug}")
        frames[slug] = df.loc[df["tumour_type"] == name].copy()
    manifest = [{"domain": domain, "cohort": slug, "tumour": "All (tumor-agnostic)" if slug == "all" else name,
                 "patients": int(frame["patient_id"].nunique()), "medication_records": len(frame)}
                for slug, frame in frames.items()
                for name in ["all" if slug == "all" else frame["tumour_type"].iloc[0]]]
    return frames, manifest


def build_from_config(cfg: dict, *, strict: bool = True) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_dir"]) / "cohorts"
    root.mkdir(parents=True, exist_ok=True)
    settings = cfg["cohorts"]
    manifest = []
    for domain, path_key in [("dataset_1", "dataset_1_csv"), ("dataset_2", "dataset_2_csv")]:
        df = pd.read_csv(cfg["paths"][path_key])
        frames, rows = construct_cohorts(
            df, domain=domain, tumour_names=settings["tumour_names"],
            min_patients=int(settings["min_patients"]),
            expected_tumours=int(settings["expected_tumour_counts"][domain]), strict=strict)
        for slug, frame in frames.items():
            frame.to_csv(root / f"{domain}_{slug}_PRIVATE.csv", index=False)
        manifest.extend(rows)
    result = pd.DataFrame(manifest)
    result.to_csv(root / "cohort_manifest_PRIVATE.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()
    print(build_from_config(load_config(args.config), strict=not args.allow_count_mismatch).to_string(index=False))

if __name__ == "__main__":
    main()
