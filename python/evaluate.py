"""Combine independent in-domain/out-of-domain bootstrap distributions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from python.common import load_config


def relative_change_bootstrap(
    in_domain: pd.DataFrame,
    out_domain: pd.DataFrame,
    patient_col: str,
    truth_col: str,
    pred_col: str,
    metrics=("accuracy", "weighted_f1"),
    n_boot=1000,
    seed=123,
):
    """Independent patient resampling; percentage change uses in-domain denominator."""
    rng = np.random.default_rng(seed)
    in_ids = in_domain[patient_col].drop_duplicates().to_numpy()
    out_ids = out_domain[patient_col].drop_duplicates().to_numpy()
    in_groups = {p: g for p, g in in_domain.groupby(patient_col, sort=False)}
    out_groups = {p: g for p, g in out_domain.groupby(patient_col, sort=False)}
    rows = []
    from sklearn.metrics import accuracy_score, f1_score

    def score(frame, metric):
        y = frame[truth_col].astype(int).to_numpy()
        p = frame[pred_col].astype(int).to_numpy()
        if metric == "accuracy":
            return accuracy_score(y, p)
        return f1_score(y, p, average="weighted", zero_division=0)

    for b in range(n_boot):
        a_ids = rng.choice(in_ids, len(in_ids), replace=True)
        b_ids = rng.choice(out_ids, len(out_ids), replace=True)
        a = pd.concat([in_groups[x] for x in a_ids], ignore_index=True)
        z = pd.concat([out_groups[x] for x in b_ids], ignore_index=True)
        row = {"iteration": b + 1}
        for metric in metrics:
            baseline = score(a, metric)
            comparison = score(z, metric)
            row[metric] = ((comparison - baseline) / baseline * 100.0) if baseline != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(config_path):
    cfg = load_config(config_path)
    out = Path(cfg["paths"]["output_dir"])
    flat = pd.read_csv(out / "dataset_1_holdout_bert_predictions_PRIVATE.csv")
    bc = pd.read_csv(out / "dataset_2_bert_predictions_PRIVATE.csv")
    # BERT: Dataset 1 in-domain; Dataset 2 out-of-domain.
    b = relative_change_bootstrap(
        flat, bc, "patient_id", "true_class", "pred_class",
        n_boot=int(cfg["analysis"]["bootstrap_iterations"]),
        seed=int(cfg["analysis"]["random_seed"]),
    )
    b.to_csv(out / "bert_relative_change_PRIVATE.csv", index=False)

    # Cyclic: Dataset 2 in-domain; Dataset 1 out-of-domain.
    cyclic_in_path = out / "dataset_2_cyclic_predictions_PRIVATE.csv"
    cyclic_out_path = out / "dataset_1_cyclic_predictions_PRIVATE.csv"
    if cyclic_in_path.exists() and cyclic_out_path.exists():
        ci = pd.read_csv(cyclic_in_path)
        co = pd.read_csv(cyclic_out_path)
        c = relative_change_bootstrap(
            ci, co, "patient_id", "true_class", "pred_class",
            n_boot=int(cfg["analysis"]["bootstrap_iterations"]),
            seed=int(cfg["analysis"]["random_seed"]) + 1,
        )
        c.to_csv(out / "cyclic_relative_change_PRIVATE.csv", index=False)

    print("Relative-change bootstrap distributions saved locally.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
