"""
Train a logistic regression model on the synthetic dataset and save it as a
pickle artifact to model_artifacts/model.pkl.

Usage:
    python data/train_model.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "synthetic_data.csv"
ARTIFACT_PATH = Path(__file__).parent.parent / "model_artifacts" / "model.pkl"

FEATURES = ["cart_size", "similar_items", "avg_price", "price_variance", "category_diversity"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Derive a binary label: high return probability = 1
    df["label"] = (df["return_prob"] >= 0.5).astype(int)

    # Add missing feature columns with sensible defaults if not present
    if "price_variance" not in df.columns:
        df["price_variance"] = 0.0
    if "category_diversity" not in df.columns:
        df["category_diversity"] = 1.0

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
