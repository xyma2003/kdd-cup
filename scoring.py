"""
Shared scoring utilities for eval scripts.

Extracted from eval_xyma.py / score_run.py / run_incremental.py to remove
triple-duplicated _normalize_value / _load_csv_values / score_task.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


def normalize_value(v: str) -> str:
    """Normalize a CSV cell value: strip whitespace, round floats to 2 decimals."""
    v = str(v).strip()
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return v
        return str(round(f, 2))
    except (ValueError, TypeError):
        return v


def load_csv_values(path: Path) -> list[tuple]:
    """Load CSV, return list of normalized row tuples (header skipped)."""
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            rows.append(tuple(normalize_value(v) for v in row))
    return rows


def score_task(prediction_path: Path, gold_path: Path) -> float:
    """
    Score = Recall - 0.5 * (Extra / Predicted)
    - Recall = matched / gold_count
    - Extra = predicted_count - matched
    """
    if not prediction_path.exists():
        return 0.0
    pred_rows = load_csv_values(prediction_path)
    gold_rows = load_csv_values(gold_path)
    if not gold_rows:
        return 1.0 if not pred_rows else 0.0
    matched = 0
    remaining_pred = list(pred_rows)
    for g_row in gold_rows:
        for i, p_row in enumerate(remaining_pred):
            if p_row == g_row or tuple(sorted(p_row)) == tuple(sorted(g_row)):
                matched += 1
                remaining_pred.pop(i)
                break
    gold_count = len(gold_rows)
    pred_count = len(pred_rows)
    recall = matched / gold_count
    extra = max(0, pred_count - matched)
    penalty = 0.5 * (extra / pred_count) if pred_count > 0 else 0.0
    return round(max(0.0, recall - penalty), 4)
