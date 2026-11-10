"""Shared analysis, metrics, and patient-level bootstrap utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
CLASS_TO_ID = {1: 0, 2: 1, 3: 2}
ID_TO_CLASS = {0: "1", 1: "2", 2: "3+"}

FEATURE_COLUMNS = [
    "tumour_type", "drug_name", "common_drug_name", "drug_category",
    "detailed_drug_category", "protocol_code", "administered_amount", "administered_units",
    "route", "ordered_date", "administered_date", "progression_date",
    "last_followup_date", "death_date",
]

DATE_COLUMNS = [
    "ordered_date", "administered_date", "progression_date",
    "last_followup_date", "death_date", "line_start_date",
]


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_columns(df: pd.DataFrame, columns: list[str], label: str = "data") -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_line(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("line_number contains missing values.")
    value = int(value)
    if value < 1:
        raise ValueError(f"LoT values must be >=1; found {value}.")
    return min(value, 3)


def _category_map(values: pd.Series) -> dict[str, int]:
    vals = sorted({str(v) for v in values.dropna().unique()})
    return {v: i for i, v in enumerate(vals)}


def fit_feature_encoders(train_df: pd.DataFrame) -> dict[str, Any]:
    """Fit categorical mappings and dose quantile boundaries on the training partition."""
    maps = {}
    categorical = [
        "tumour_type", "drug_name", "common_drug_name", "drug_category",
        "detailed_drug_category", "protocol_code", "administered_units", "route",
    ]
    for col in categorical:
        maps[col] = _category_map(train_df[col].astype("string")) if col in train_df else {}
    amount = pd.to_numeric(train_df["administered_amount"], errors="coerce").dropna()
    if amount.empty:
        bins = []
    else:
        _, edges = pd.qcut(amount, q=5, retbins=True, duplicates="drop")
        bins = sorted(set(float(x) for x in edges))
        if len(bins) < 2:
            bins = []
    return {"category_maps": maps, "amount_bin_edges": bins}


def _category_token(value: Any, prefix: str, mapping: dict[str, int]) -> str:
    if pd.isna(value):
        return f"{prefix}UNK"
    key = str(value)
    return f"{prefix}{mapping.get(key, 'UNK')}"


def _date_token(value: Any, prefix: str) -> str:
    if pd.isna(value):
        return f"{prefix}MISSING"
    return f"{prefix}{pd.Timestamp(value).to_period('M')}"


def build_record_tokens(df: pd.DataFrame, encoders: dict[str, Any]) -> list[str]:
    """Create one composite token per medication record, using training-fitted mappings."""
    maps = encoders["category_maps"]
    edges = encoders["amount_bin_edges"]
    out = []
    for row in df.itertuples(index=False):
        r = row._asdict()
        amount = pd.to_numeric(pd.Series([r.get("administered_amount")]), errors="coerce").iloc[0]
        if pd.isna(amount) or len(edges) < 2:
            amount_token = "AMT_UNK"
        else:
            b = int(np.clip(np.digitize(float(amount), edges[1:-1], right=False), 0, max(len(edges)-2, 0)))
            amount_token = f"AMT_{b}"
        parts = [
            _category_token(r.get("tumour_type"), "TUMOUR_", maps["tumour_type"]),
            _category_token(r.get("drug_name"), "DRUG_", maps["drug_name"]),
            _category_token(r.get("common_drug_name"), "COMMONDRUG_", maps["common_drug_name"]),
            _category_token(r.get("drug_category"), "DRUGCAT_", maps["drug_category"]),
            _category_token(r.get("detailed_drug_category"), "DETAILCAT_", maps["detailed_drug_category"]),
            _category_token(r.get("protocol_code"), "PROTOCOL_", maps["protocol_code"]),
            amount_token,
            _category_token(r.get("administered_units"), "UNIT_", maps["administered_units"]),
            _category_token(r.get("route"), "ROUTE_", maps["route"]),
            _date_token(r.get("ordered_date"), "ORDER_"),
            _date_token(r.get("administered_date"), "ADMIN_"),
            _date_token(r.get("progression_date"), "PROG_"),
            _date_token(r.get("last_followup_date"), "LASTFU_"),
            _date_token(r.get("death_date"), "DEATH_"),
        ]
        out.append("__".join(parts))
    return out


def prepare_patient_sequences(
    df: pd.DataFrame,
    encoders: dict[str, Any],
    max_length: int,
    patient_col: str = "patient_id",
) -> tuple[list[list[str]], list[list[int]], list[Any]]:
    required = [patient_col, "line_number"] + FEATURE_COLUMNS
    require_columns(df, required)
    work = df.copy()
    for col in DATE_COLUMNS:
        if col in work:
            work[col] = pd.to_datetime(work[col], errors="coerce")
    work["line_class"] = work["line_number"].map(normalize_line)
    sort_cols = [patient_col]
    if "administered_date" in work.columns and work["administered_date"].notna().any():
        work["_sequence_date"] = work["administered_date"].fillna(work["ordered_date"])
    else:
        work["_sequence_date"] = work["ordered_date"]
    sort_cols.append("_sequence_date")
    work = work.sort_values(sort_cols, kind="stable", na_position="last")

    sequences, labels, patient_ids = [], [], []
    for pid, group in work.groupby(patient_col, sort=False, dropna=False):
        tokens = build_record_tokens(group, encoders)
        labs = [CLASS_TO_ID[int(x)] for x in group["line_class"].tolist()]
        sequences.append(tokens[:max_length])
        labels.append(labs[:max_length])
        patient_ids.append(pid)
    return sequences, labels, patient_ids


def fit_wordlevel_tokenizer(sequences: list[list[str]]):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.WordLevelTrainer(special_tokens=SPECIAL_TOKENS)
    tok.train_from_iterator((" ".join(s) for s in sequences), trainer)
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    return hf


def encode_sequences(tokenizer, sequences, labels, max_length):
    enc = tokenizer(
        sequences,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    padded = [list(y[:max_length]) + [-100] * (max_length-len(y[:max_length])) for y in labels]
    for i, y in enumerate(labels):
        n_nonpad = int(sum(v != tokenizer.pad_token_id for v in enc["input_ids"][i]))
        if n_nonpad != min(len(y), max_length):
            raise ValueError(
                "Token/label alignment check failed. A medication record did not map to exactly one token."
            )
    return enc, padded


def flatten_valid(logits: np.ndarray, labels: np.ndarray):
    pred = logits.argmax(axis=-1)
    mask = labels != -100
    return labels[mask].astype(int), pred[mask].astype(int), logits[mask]


def compute_metrics(y_true, y_pred, probabilities=None, include_auroc=False):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if include_auroc:
        if probabilities is None:
            raise ValueError("Probabilities are required for AUROC.")
        per_class = []
        for cls in range(3):
            binary = (np.asarray(y_true) == cls).astype(int)
            if binary.min() == binary.max():
                return {**result, "macro_auroc": float("nan")}
            per_class.append(roc_auc_score(binary, probabilities[:, cls]))
        result["macro_auroc"] = float(np.mean(per_class))
    return result


def patient_bootstrap(
    records: pd.DataFrame,
    patient_col: str,
    truth_col: str,
    pred_col: str,
    probability_cols: list[str] | None = None,
    n_boot: int = 1000,
    seed: int = 123,
    include_auroc: bool = False,
) -> pd.DataFrame:
    """Resample patients, preserving all records and multiplicity of sampled patients."""
    require_columns(records, [patient_col, truth_col, pred_col])
    rng = np.random.default_rng(seed)
    ids = records[patient_col].drop_duplicates().to_numpy()
    if len(ids) == 0:
        raise ValueError("No patients available for bootstrap.")
    rows = []
    grouped = {pid: g for pid, g in records.groupby(patient_col, sort=False)}
    for b in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        chunks = [grouped[pid] for pid in sampled]
        boot = pd.concat(chunks, ignore_index=True)
        y = boot[truth_col].astype(int).to_numpy()
        p = boot[pred_col].astype(int).to_numpy()
        probs = boot[probability_cols].to_numpy(dtype=float) if probability_cols else None
        metrics = compute_metrics(y, p, probs, include_auroc)
        rows.append({"iteration": b + 1, **metrics})
    return pd.DataFrame(rows)


def percentile_summary(point_metrics: dict, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, point in point_metrics.items():
        vals = boot[metric].dropna()
        rows.append({
            "metric": metric,
            "estimate": float(point),
            "ci_lower": float(vals.quantile(0.025)) if len(vals) else float("nan"),
            "ci_upper": float(vals.quantile(0.975)) if len(vals) else float("nan"),
            "valid_bootstrap_iterations": int(len(vals)),
        })
    return pd.DataFrame(rows)


def save_json(obj: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
