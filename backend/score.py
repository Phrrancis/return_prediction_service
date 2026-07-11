"""
Live scoring worker (shadow mode) + outcome maturation job.

score_pending()  : scores every cart lacking a prediction. Async by design —
                   shadow mode has no latency budget, so a poll loop is enough.
mature_outcomes(): nightly job that finalizes labels once return window + buffer
                   has elapsed. Predictions are append-only; outcomes only exist
                   past maturity. These two invariants are the report's honesty.
"""

from __future__ import annotations

import json

import numpy as np
import shap

from backend.db import get_conn
from backend.features import FEATURE_NAMES, build_features, humanize
from backend.train import load_model


def score_pending(merchant_id: str, db_path: str | None = None, limit: int = 50000) -> int:
    art = load_model(merchant_id)
    model = art["model"]
    explainer = shap.TreeExplainer(model)

    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.* FROM carts c
            LEFT JOIN predictions p
              ON p.cart_token = c.cart_token AND p.merchant_id = c.merchant_id
            WHERE c.merchant_id = ? AND p.id IS NULL
            LIMIT ?
            """,
            (merchant_id, limit),
        ).fetchall()

        scored = 0
        for r in rows:
            row = dict(r)
            vec, _ = build_features(row, art["sku_stats"], art["user_stats"], art["base_rate"])
            X = np.array([vec], dtype=float)
            score = float(model.predict_proba(X)[0, 1])

            sv = explainer.shap_values(X)
            sv = sv[1][0] if isinstance(sv, list) else sv[0]
            top_idx = np.argsort(-np.abs(sv))[:3]
            explanations = [
                humanize(FEATURE_NAMES[i], vec[i]) for i in top_idx if abs(sv[i]) > 1e-6
            ]

            conn.execute(
                """INSERT INTO predictions
                   (merchant_id, cart_token, model_version, score, explanations)
                   VALUES (?,?,?,?,?)""",
                (merchant_id, row["cart_token"], art["version"], score, json.dumps(explanations)),
            )
            scored += 1
    return scored


def mature_outcomes(merchant_id: str, db_path: str | None = None) -> int:
    """In production: check orders past (return window + logistics buffer),
    finalize returned/kept from refund & return webhooks. The demo seeds
    outcomes directly, so this verifies the invariant instead."""
    with get_conn(db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM outcomes WHERE merchant_id=? AND matured_at <= datetime('now')",
            (merchant_id,),
        ).fetchone()[0]
    return n
