"""Train Dataset 1 BERT using a patient-level 80/20 split and training-only CV."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (
    BertConfig, BertForTokenClassification, Trainer, TrainingArguments,
    EarlyStoppingCallback,
)

from python.common import (
    load_config, fit_feature_encoders, prepare_patient_sequences,
    fit_wordlevel_tokenizer, encode_sequences, flatten_valid, save_json,
)

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def subset_encoding(enc, indices):
    return {k: [v[i] for i in indices] for k, v in enc.items()}


def main(config_path):
    cfg = load_config(config_path)
    seed = int(cfg["analysis"]["random_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out = Path(cfg["paths"]["output_dir"])
    model_dir = out / "bert_dataset_1_model"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["paths"]["dataset_1_csv"])
    patient_col = cfg["features"]["patient_id"]
    label_col = cfg["features"]["line_number"]
    if df[patient_col].isna().any():
        raise ValueError("Patient identifiers cannot be missing.")
    df = df.rename(columns={v: k for k, v in cfg["features"].items() if v in df.columns}).copy()
    patient_col, label_col = "patient_id", "line_number"
    df["_line_collapsed"] = pd.to_numeric(df[label_col], errors="raise").clip(upper=3).astype(int)

    # Patient-level stratification by maximum collapsed LoT class.
    patient_strata = df.groupby(patient_col)["_line_collapsed"].max().sort_index()
    patient_ids = patient_strata.index.to_numpy()
    strata = patient_strata.to_numpy()
    train_ids, holdout_ids = train_test_split(
        patient_ids,
        test_size=float(cfg["analysis"]["test_size"]),
        random_state=seed,
        stratify=strata,
    )
    train_df = df[df[patient_col].isin(train_ids)].copy()
    holdout_df = df[df[patient_col].isin(holdout_ids)].copy()

    feature_encoders = fit_feature_encoders(train_df)
    train_seq, train_y, train_pid = prepare_patient_sequences(
        train_df, feature_encoders, int(cfg["analysis"]["max_length"]), patient_col
    )
    hold_seq, hold_y, hold_pid = prepare_patient_sequences(
        holdout_df, feature_encoders, int(cfg["analysis"]["max_length"]), patient_col
    )
    tokenizer = fit_wordlevel_tokenizer(train_seq)
    train_enc, train_labels = encode_sequences(
        tokenizer, train_seq, train_y, int(cfg["analysis"]["max_length"])
    )
    hold_enc, hold_labels = encode_sequences(
        tokenizer, hold_seq, hold_y, int(cfg["analysis"]["max_length"])
    )

    split_path = out / "dataset_1_patient_split_PRIVATE.csv"
    pd.DataFrame({
        "patient_id": list(train_ids) + list(holdout_ids),
        "partition": ["train_tuning"] * len(train_ids) + ["evaluation"] * len(holdout_ids),
    }).to_csv(split_path, index=False)

    train_patient_strata = (
        train_df.groupby(patient_col)["_line_collapsed"].max()
        .reindex(train_pid).to_numpy() - 1
    )
    if len(np.unique(train_patient_strata)) < 2:
        raise ValueError("At least two patient-level LoT strata are required for CV.")
    n_splits = int(cfg["analysis"]["cv_folds"])
    if np.min(np.unique(train_patient_strata, return_counts=True)[1]) < n_splits:
        raise ValueError("A patient-level LoT stratum has fewer patients than CV folds.")

    model_cfg = cfg["model"]
    base_config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=int(model_cfg["hidden_size"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        num_attention_heads=int(model_cfg["num_attention_heads"]),
        intermediate_size=int(model_cfg["intermediate_size"]),
        num_labels=3,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        y_true, y_pred, _ = flatten_valid(logits, labels)
        return {
            "accuracy": float((y_true == y_pred).mean()),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    def objective(trial):
        lr = trial.suggest_float(
            "learning_rate", float(model_cfg["learning_rate_min"]),
            float(model_cfg["learning_rate_max"]), log=True
        )
        batch_size = trial.suggest_categorical("per_device_train_batch_size", model_cfg["batch_sizes"])
        epochs = trial.suggest_int(
            "num_train_epochs", int(model_cfg["epochs_min"]), int(model_cfg["epochs_max"])
        )
        weight_decay = trial.suggest_float(
            "weight_decay", float(model_cfg["weight_decay_min"]),
            float(model_cfg["weight_decay_max"])
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_scores = []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(np.zeros(len(train_pid)), train_patient_strata)):
            tr_ds = TokenDataset(subset_encoding(train_enc, tr_idx), [train_labels[i] for i in tr_idx])
            val_ds = TokenDataset(subset_encoding(train_enc, val_idx), [train_labels[i] for i in val_idx])
            args = TrainingArguments(
                output_dir=str(out / "optuna_tmp" / f"trial_{trial.number}_fold_{fold}"),
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                per_device_train_batch_size=int(batch_size),
                per_device_eval_batch_size=int(batch_size),
                num_train_epochs=int(epochs),
                learning_rate=lr,
                weight_decay=weight_decay,
                seed=seed,
                data_seed=seed,
                report_to="none",
                save_total_limit=1,
                logging_dir=None,
            )
            trainer = Trainer(
                model_init=lambda: BertForTokenClassification(base_config),
                args=args,
                train_dataset=tr_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=int(cfg["analysis"]["early_stopping_patience"])
                )],
            )
            trainer.train()
            fold_scores.append(float(trainer.state.best_metric))
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(cfg["analysis"]["optuna_trials"]))
    best = study.best_params
    save_json({"best_value_weighted_f1": study.best_value, "best_params": best},
              out / "bert_hyperparameter_search.json")

    train_ds = TokenDataset(train_enc, train_labels)
    hold_ds = TokenDataset(hold_enc, hold_labels)
    final_args = TrainingArguments(
        output_dir=str(out / "final_training"),
        eval_strategy="no",
        save_strategy="no",
        per_device_train_batch_size=int(best["per_device_train_batch_size"]),
        num_train_epochs=int(best["num_train_epochs"]),
        learning_rate=float(best["learning_rate"]),
        weight_decay=float(best["weight_decay"]),
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    final_trainer = Trainer(
        model=BertForTokenClassification(base_config),
        args=final_args,
        train_dataset=train_ds,
    )
    final_trainer.train()
    model_dir.mkdir(parents=True, exist_ok=True)
    final_trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_json({
        "feature_encoders": feature_encoders,
        "class_to_id": {"1": 0, "2": 1, "3+": 2},
        "max_length": int(cfg["analysis"]["max_length"]),
        "patient_column": patient_col,
        "training_patient_count": len(train_ids),
        "holdout_patient_count": len(holdout_ids),
    }, model_dir / "model_metadata.json")

    pred_output = final_trainer.predict(hold_ds)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    rows = []
    for idx, pid in enumerate(hold_pid):
        n = len(hold_y[idx])
        for j in range(n):
            rows.append({
                "patient_id": pid,
                "true_class": int(hold_y[idx][j]),
                "pred_class": int(np.argmax(logits[idx, j])),
                "prob_class_1": float(probs[idx, j, 0]),
                "prob_class_2": float(probs[idx, j, 1]),
                "prob_class_3plus": float(probs[idx, j, 2]),
            })
    pd.DataFrame(rows).to_csv(out / "dataset_1_holdout_bert_predictions_PRIVATE.csv", index=False)
    print(f"Training complete. Private outputs written to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
