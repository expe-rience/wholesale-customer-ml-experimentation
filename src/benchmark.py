from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "Region", "Fresh", "Milk", "Grocery", "Frozen",
    "Detergents_Paper", "Delicassen",
]
TARGET = "class"
NUMERIC = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
CATEGORICAL = ["Region"]


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Fit all transforms inside CV folds; avoid preprocessing leakage."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


def get_models():
    """Return (estimator, requires_scaling) pairs."""
    models = {
        "Dummy": (DummyClassifier(strategy="most_frequent"), False),
        "LogisticRegression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"), True
        ),
        "KNN": (KNeighborsClassifier(n_neighbors=5, weights="distance"), True),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42, class_weight="balanced"), False
        ),
        "Bagging": (
            BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=100, random_state=42, n_jobs=-1,
            ), False
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=300, random_state=42,
                class_weight="balanced", n_jobs=-1,
            ), False
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(
                n_estimators=300, random_state=42,
                class_weight="balanced", n_jobs=-1,
            ), False
        ),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), False),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=42), False
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=-1,
            ), False
        )
    except ImportError:
        print("Optional dependency missing: XGBoost. Install requirements-boosting.txt.")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = (
            LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=15,
                class_weight="balanced", random_state=42,
                verbosity=-1, n_jobs=-1,
            ), False
        )
    except ImportError:
        print("Optional dependency missing: LightGBM. Install requirements-boosting.txt.")

    return models


def load_dataset(path: str | Path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[FEATURES + [TARGET]].dropna(subset=[TARGET]).copy()
    return df[FEATURES], df[TARGET]


def benchmark(data_path="data/multiclass.csv", output_dir="artifacts"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(data_path)
    if y.nunique() < 2:
        raise ValueError("At least two target classes are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2:
        raise ValueError("Not enough examples in the smallest class for stratified CV.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    rows, candidates = [], {}
    for name, (estimator, scale_numeric) in get_models().items():
        pipe = Pipeline([
            ("preprocess", make_preprocessor(scale_numeric)),
            ("model", estimator),
        ])
        try:
            start = time.perf_counter()
            scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1
            )
            cv_seconds = time.perf_counter() - start

            # Measure a training fit on training data only. Do not inspect test
            # performance for every candidate; reserve the test set for the winner.
            start = time.perf_counter()
            pipe.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start

            row = {
                "model": name,
                "cv_macro_f1_mean": float(scores.mean()),
                "cv_macro_f1_std": float(scores.std()),
                "cv_seconds": float(cv_seconds),
                "fit_seconds": float(fit_seconds),
            }
            rows.append(row)
            candidates[name] = pipe
            print(f"{name}: CV macro-F1={scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as exc:
            print(f"Skipping {name}: {type(exc).__name__}: {exc}")

    if not rows:
        raise RuntimeError("No models completed successfully.")

    ranking = pd.DataFrame(rows).sort_values(
        ["cv_macro_f1_mean", "cv_macro_f1_std"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Select using CV only, then evaluate exactly this selected candidate on test.
    winner_name = str(ranking.iloc[0]["model"])
    winner = candidates[winner_name]
    predict_start = time.perf_counter()
    predictions = winner.predict(X_test)
    predict_seconds = time.perf_counter() - predict_start

    winner_row = ranking.iloc[0].to_dict()
    winner_metrics = {
        "cv_macro_f1_mean": float(winner_row["cv_macro_f1_mean"]),
        "cv_macro_f1_std": float(winner_row["cv_macro_f1_std"]),
        "test_macro_f1": float(f1_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_prediction_seconds": float(predict_seconds),
    }

    ranking.to_csv(output_dir / "model_comparison.csv", index=False)
    joblib.dump(winner, output_dir / "best_model.joblib")

    labels = list(winner.classes_)
    cm = confusion_matrix(y_test, predictions, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    summary = {
        "selected_model": winner_name,
        "selection_metric": "cross-validation macro-F1",
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        **winner_metrics,
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    (output_dir / "best_model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nRanking by cross-validation macro-F1:")
    print(ranking.to_string(index=False))
    print(f"\nSelected model: {winner_name}")
    print("\nHeld-out test classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    return ranking


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/multiclass.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    benchmark(args.data, args.output)
