#!/usr/bin/env python3
"""
Quick exploratory script for SIGNATE train.csv data.

Loads train.csv, prints dataset shape, missing-value summary, numerical
descriptive statistics, and the target balance distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"[analysis] pandas import failed: {exc}", file=sys.stderr)
    sys.exit(1)

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"


def load_train(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[analysis] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if df.empty:
        print(f"[analysis] train.csv has no rows.", file=sys.stderr)
        sys.exit(1)
    return df


def show_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"[analysis] Dataset shape -> rows: {rows:,} | columns: {cols}")


def show_missing(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[analysis] Missing values -> none detected.")
        return
    print("[analysis] Missing values per column:")
    for col, count in missing.sort_values(ascending=False).items():
        ratio = count / len(df)
        print(f"          {col:>12} : {count:>6} ({ratio:.2%})")


def show_numeric_summary(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        print("[analysis] Numeric summary -> no numeric columns detected.")
        return
    summary = numeric_df.describe().transpose()
    print("[analysis] Numeric summary (describe):")
    print(summary.to_string(float_format=lambda v: f"{v:,.2f}"))


def show_target_balance(df: pd.DataFrame, target: str = "y") -> None:
    if target not in df.columns:
        print(f"[analysis] Target column '{target}' not found.")
        return
    counts = df[target].value_counts(dropna=False).sort_index()
    print(f"[analysis] Target balance for '{target}':")
    total = counts.sum()
    for value, count in counts.items():
        ratio = count / total if total else 0.0
        print(f"          {value!r:>6} : {count:>6} ({ratio:.2%})")


def show_target_numeric_corr(df: pd.DataFrame, target: str = "y") -> None:
    if target not in df.columns:
        return
    try:
        target_numeric = pd.to_numeric(df[target], errors="coerce")
    except Exception:  # pragma: no cover - defensive
        return
    if target_numeric.isna().all():
        return
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=[target], errors="ignore")
    if numeric_df.empty:
        return
    correlations = numeric_df.corrwith(target_numeric)
    correlations = correlations.dropna()
    if correlations.empty:
        return
    print(f"[analysis] Pearson correlation with '{target}':")
    for col, corr in correlations.sort_values(key=lambda s: s.abs(), ascending=False).items():
        print(f"          {col:>12} : {corr:+.3f}")


def main(path: Optional[Path] = None) -> None:
    csv_path = path or TRAIN_PATH
    df = load_train(csv_path)
    show_shape(df)
    show_missing(df)
    show_numeric_summary(df)
    show_target_balance(df, target="y")
    show_target_numeric_corr(df, target="y")


if __name__ == "__main__":
    custom_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(custom_path)
