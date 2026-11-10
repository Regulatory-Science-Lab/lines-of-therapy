"""Apply the trained Dataset 1 BERT model to Dataset 2 or Dataset 1 holdout records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertForTokenClassification, PreTrainedTokenizerFast

from python.common import (
    load_config, prepare_patient_sequences, encode_sequences,
    patient_bootstrap, compute_metrics, percentile_summary,
)

def main(config_path, dataset):
    cfg = load_config(config_path)
    out = Path(cfg["paths"]["output_dir"])
    model_dir = out / "bert_dataset_1_model"
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = BertForTokenClassification.from_pretrained(model_dir)
    model.eval()

    if dataset == "dataset_2":
        data_path = cfg["paths"]["dataset_2_csv"]
        df = pd.read_csv(data_path)
    elif dataset == "dataset_1_holdout":
        data_path = cfg["paths"]["dataset_1_csv"]
        df = pd.read_csv(data_path)
        split = pd.read_csv(out / "dataset_1_patient_split_PRIVATE.csv")
        ids = split.loc[split["partition"] == "evaluation", "patient_id"]
        df = df[df[cfg["features"]["patient_id"]].isin(ids)].copy()
    else:
        raise ValueError("dataset must be 'dataset_2' or 'dataset_1_holdout'.")

    patient_col = cfg["features"]["patient_id"]
    df = df.rename(columns={v: k for k, v in cfg["features"].items() if v in df.columns})
    seq, labels, patient_ids = prepare_patient_sequences(
        df, metadata["feature_encoders"], int(metadata["max_length"]), patient_col="patient_id"
    )
    enc, padded_labels = encode_sequences(tokenizer, seq, labels, int(metadata["max_length"]))

    rows = []
    batch_size = int(cfg["model"].get("inference_batch_size", 16))
    for start in range(0, len(seq), batch_size):
        stop = min(start + batch_size, len(seq))
        batch = {k: torch.tensor(v[start:stop], dtype=torch.long) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**batch).logits.cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        for local_i, pid in enumerate(patient_ids[start:stop]):
            n = len(labels[start + local_i])
            for j in range(n):
                rows.append({
                    "patient_id": pid,
                    "true_class": int(labels[start + local_i][j]),
                    "pred_class": int(np.argmax(logits[local_i, j])),
                    "prob_class_1": float(probs[local_i, j, 0]),
                    "prob_class_2": float(probs[local_i, j, 1]),
                    "prob_class_3plus": float(probs[local_i, j, 2]),
                })

    results = pd.DataFrame(rows)
    tag = "dataset_1_holdout" if dataset == "dataset_1_holdout" else "dataset_2"
    private_csv = out / f"{tag}_bert_predictions_PRIVATE.csv"
    results.to_csv(private_csv, index=False)
    prob_cols = ["prob_class_1", "prob_class_2", "prob_class_3plus"]
    point = compute_metrics(
        results["true_class"].to_numpy(), results["pred_class"].to_numpy(),
        results[prob_cols].to_numpy(), include_auroc=True
    )
    boot = patient_bootstrap(
        results, "patient_id", "true_class", "pred_class",
        probability_cols=prob_cols, n_boot=int(cfg["analysis"]["bootstrap_iterations"]),
        seed=int(cfg["analysis"]["random_seed"]), include_auroc=True,
    )
    summary = percentile_summary(point, boot)
    summary.insert(0, "dataset", tag)
    summary.to_csv(out / f"{tag}_bert_metrics_PRIVATE.csv", index=False)
    boot.to_csv(out / f"{tag}_bert_bootstrap_PRIVATE.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", choices=["dataset_2", "dataset_1_holdout"], required=True)
    a = p.parse_args()
    main(a.config, a.dataset)
