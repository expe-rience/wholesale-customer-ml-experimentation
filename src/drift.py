"""Input-drift checks: compare new data against the training (reference) data.

Run:
    python -m src.drift --reference data/multiclass.csv --current new_customers.csv

Population Stability Index (PSI) rule of thumb:
    < 0.10  no meaningful shift
    0.10-0.25  moderate shift, investigate
    > 0.25  major shift, consider retraining
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.benchmark import NUMERIC

PSI_WARN = 0.10
PSI_ALERT = 0.25


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the reference data."""
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur = pd.Series(current).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("Both samples need at least one non-null value.")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # constant or near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6  # avoid log(0) for empty bins
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """Two-sample Kolmogorov-Smirnov distance (0 = identical, 1 = disjoint)."""
    ref = np.sort(pd.Series(reference).dropna().to_numpy(dtype=float))
    cur = np.sort(pd.Series(current).dropna().to_numpy(dtype=float))
    values = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, values, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, values, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def check_drift(reference: pd.DataFrame, current: pd.DataFrame,
                features: list[str] | None = None) -> pd.DataFrame:
    """Return one row per feature with PSI, KS distance and a status."""
    features = features or NUMERIC
    rows = []
    for col in features:
        value = psi(reference[col], current[col])
        status = "alert" if value > PSI_ALERT else "warn" if value > PSI_WARN else "ok"
        rows.append({
            "feature": col,
            "psi": round(value, 4),
            "ks_distance": round(ks_statistic(reference[col], current[col]), 4),
            "status": status,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check input drift vs. training data.")
    parser.add_argument("--reference", default="data/multiclass.csv")
    parser.add_argument("--current", required=True)
    args = parser.parse_args()
    report = check_drift(pd.read_csv(args.reference), pd.read_csv(args.current))
    print(report.to_string(index=False))
    if (report["status"] == "alert").any():
        raise SystemExit("Drift alert: consider investigating or retraining.")
