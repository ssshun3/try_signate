"""
Quick validation/analysis helper for the SIGNATE bank marketing dataset.

Usage:
    python validation_analysis.py

Outputs high-level profile information for both train.csv and test.csv so that
feature engineering and modeling notebooks can focus on the interesting parts.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd

    _HAVE_PANDAS = True
except Exception as exc:  # pragma: no cover - optional dependency
    _HAVE_PANDAS = False
    _PANDAS_ERR = exc

DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    unique: int
    top_values: List[Tuple[str, int]]
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


def _print(msg: str) -> None:
    print(f"[validation] {msg}")


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _is_float(value: str) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def summarize_with_csv(path: Path, max_top: int = 5) -> List[ColumnSummary]:
    _print(f"Profiling via csv module -> {path}")
    rows = list(_iter_csv_rows(path))
    if not rows:
        return []

    columns = rows[0].keys()
    summaries: List[ColumnSummary] = []

    for col in columns:
        values = [row[col] for row in rows]
        non_null_values = [v for v in values if v not in ("", None)]
        missing = len(values) - len(non_null_values)
        unique = len(set(non_null_values))

        counter = Counter(non_null_values)
        top_values = counter.most_common(max_top)

        numeric_values = [float(v) for v in non_null_values if _is_float(v)]
        if numeric_values:
            mean = statistics.fmean(numeric_values)
            std = statistics.pstdev(numeric_values) if len(numeric_values) > 1 else 0.0
            summaries.append(
                ColumnSummary(
                    name=col,
                    dtype="numeric",
                    non_null=len(non_null_values),
                    missing=missing,
                    unique=unique,
                    top_values=top_values,
                    mean=mean,
                    std=std,
                    min=min(numeric_values),
                    max=max(numeric_values),
                )
            )
        else:
            summaries.append(
                ColumnSummary(
                    name=col,
                    dtype="categorical",
                    non_null=len(non_null_values),
                    missing=missing,
                    unique=unique,
                    top_values=top_values,
                )
            )

    return summaries


def summarize_with_pandas(path: Path, max_top: int = 5) -> List[ColumnSummary]:
    if not _HAVE_PANDAS:
        raise RuntimeError(f"Pandas not available: {_PANDAS_ERR}")

    _print(f"Profiling via pandas -> {path}")
    df = pd.read_csv(path)

    summaries: List[ColumnSummary] = []
    for col in df.columns:
        series = df[col]
        non_null = series.notna().sum()
        missing = series.isna().sum()
        unique = series.nunique(dropna=True)
        top_values = list(series.value_counts(dropna=True).head(max_top).items())

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            summaries.append(
                ColumnSummary(
                    name=col,
                    dtype=str(series.dtype),
                    non_null=int(non_null),
                    missing=int(missing),
                    unique=int(unique),
                    top_values=[(str(k), int(v)) for k, v in top_values],
                    mean=float(desc["mean"]) if not math.isnan(desc["mean"]) else None,
                    std=float(desc["std"]) if not math.isnan(desc["std"]) else None,
                    min=float(desc["min"]) if not math.isnan(desc["min"]) else None,
                    max=float(desc["max"]) if not math.isnan(desc["max"]) else None,
                )
            )
        else:
            summaries.append(
                ColumnSummary(
                    name=col,
                    dtype=str(series.dtype),
                    non_null=int(non_null),
                    missing=int(missing),
                    unique=int(unique),
                    top_values=[(str(k), int(v)) for k, v in top_values],
                )
            )

    return summaries


def summarize_file(path: Path) -> List[ColumnSummary]:
    if not path.exists():
        _print(f"File not found: {path}")
        return []

    if _HAVE_PANDAS:
        try:
            return summarize_with_pandas(path)
        except Exception as exc:
            _print(f"Pandas profiling failed ({exc}); falling back to csv module.")

    return summarize_with_csv(path)


def show_summary(name: str, summaries: List[ColumnSummary]) -> None:
    if not summaries:
        _print(f"No data summarised for {name}.")
        return

    _print(f"--- {name} ({len(summaries)} columns) ---")
    for summary in summaries:
        _print(
            f"{summary.name:>12} | {summary.dtype:<12} | "
            f"non-null={summary.non_null:>6} | missing={summary.missing:>6} | "
            f"unique={summary.unique:>6}"
        )
        if summary.mean is not None:
            _print(
                f"{'':>12} | stats -> mean={summary.mean:.3f} std={summary.std:.3f} "
                f"min={summary.min:.3f} max={summary.max:.3f}"
            )
        if summary.top_values:
            tops = ", ".join(f"{val} ({cnt})" for val, cnt in summary.top_values)
            _print(f"{'':>12} | top -> {tops}")


def class_balance(train_summary: List[ColumnSummary], target_col: str = "y") -> None:
    target_info = next((s for s in train_summary if s.name == target_col), None)
    if not target_info:
        _print("Target column 'y' not found; skip balance check.")
        return

    if not target_info.top_values:
        _print("No target distribution available.")
        return

    total = sum(count for _, count in target_info.top_values)
    balance = ", ".join(
        f"{val}: {count} ({count / total:.2%})" for val, count in target_info.top_values
    )
    _print(f"Target balance -> {balance}")


def main() -> None:
    _print("Starting dataset validation summary ...")
    train_summary = summarize_file(TRAIN_PATH)
    test_summary = summarize_file(TEST_PATH)

    show_summary("train.csv", train_summary)
    class_balance(train_summary, target_col="y")
    show_summary("test.csv", test_summary)

    _print("Done.")


if __name__ == "__main__":
    main()
