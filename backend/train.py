"""
Per-merchant training pipeline.

Key discipline: TEMPORAL split, never random. SKU return-rate features are
computed from *training-window data only*, then the model is validated on the
later hold-out months it has never seen. A random split would leak future
return rates backward in time and inflate every metric you show a merchant.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime

import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from backend.db import REPO_ROOT, get_conn
from backend.features import FEATURE_NAMES, build_features

logger = logging.getLogger(__name__)

MODELS_DIR = REPO_ROOT / "models"


def compute_stats(rows: list[dict]) -> tuple[dict, dict]:
    """SKU / category / user aggregates from a set of (cart, outcome) rows."""
    sku_stats: dict = {}
    user_stats: dict = {}
    for r in rows:
        items = json.loads(r["line_items"])
        returned = r["returned"]
        for it in items:
            s = sku_stats.setdefault(it["sku"], {"orders": 0, "returns": 0})
            s["orders"] += 1
            s["returns"] += returned
            cat = sku_stats.setdefault(
                f"__category__{it.get('category', 'unknown')}", {"orders": 0, "returns": 0}
            )
            cat["orders"] += 1
            cat["returns"] += returned
        if r["customer_id"]:
            u = user_stats.setdefault(
                r["customer_id"], {"orders": 0, "returns": 0, "last_order_days": 0.0}
            )
            u["orders"] += 1
            u["returns"] += returned
    return sku_stats, user_stats


def load_labeled_carts(merchant_id: str, db_path: str | None = None) -> list[dict]:
    """Carts joined to mature outcomes — the training corpus."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.cart_token, c.line_items, c.customer_id, c.payment, c.created_at,
                   o.returned
            FROM carts c
            JOIN outcomes o ON o.order_id = c.order_id AND o.merchant_id = c.merchant_id
            WHERE c.merchant_id = ?
              AND o.matured_at <= datetime('now')
            ORDER BY c.created_at
            """,
            (merchant_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def train(merchant_id: str, db_path: str | None = None, train_frac: float = 0.75):
    rows = load_labeled_carts(merchant_id, db_path)
    if len(rows) < 500:
        raise ValueError(f"Only {len(rows)} labeled carts; need >=500 to train sensibly.")

    split = int(len(rows) * train_frac)          # rows are time-ordered
    train_rows, valid_rows = rows[:split], rows[split:]

    # stats from the TRAINING window only — no leakage into validation
    sku_stats, user_stats = compute_stats(train_rows)
    base_rate = sum(r["returned"] for r in train_rows) / len(train_rows)

    def to_xy(subset):
        X = [build_features(r, sku_stats, user_stats, base_rate)[0] for r in subset]
        y = [r["returned"] for r in subset]
        return np.array(X, dtype=float), np.array(y)

    X_tr, y_tr = to_xy(train_rows)
    X_va, y_va = to_xy(valid_rows)

    model = lgb.LGBMClassifier(
        n_estimators=250, max_depth=7, learning_rate=0.05,
        num_leaves=63, min_child_samples=30,
        random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(25, verbose=False)])

    proba = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, proba)
    prec, rec, thr = precision_recall_curve(y_va, proba)

    # operating points a merchant can reason about
    op_points = {}
    for target_p in (0.60, 0.70, 0.80):
        idx = np.argmax(prec[:-1] >= target_p) if (prec[:-1] >= target_p).any() else None
        if idx is not None:
            op_points[f"precision_{int(target_p * 100)}"] = {
                "threshold": float(thr[idx]),
                "precision": float(prec[idx]),
                "recall": float(rec[idx]),
            }

    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "model": model, "sku_stats": sku_stats, "user_stats": user_stats,
        "base_rate": base_rate, "feature_names": FEATURE_NAMES,
        "metrics": {"auc": auc, "n_train": len(train_rows),
                    "n_valid": len(valid_rows), "operating_points": op_points},
        "version": version, "merchant_id": merchant_id,
    }
    with open(MODELS_DIR / f"{merchant_id}_{version}.pkl", "wb") as f:
        pickle.dump(artifact, f)
    with open(MODELS_DIR / f"{merchant_id}_latest.pkl", "wb") as f:
        pickle.dump(artifact, f)

    logger.info("Trained model for merchant=%s version=%s auc=%.3f", merchant_id, version, auc)
    return artifact["metrics"], version


def load_model(merchant_id: str) -> dict:
    with open(MODELS_DIR / f"{merchant_id}_latest.pkl", "rb") as f:
        return pickle.load(f)
