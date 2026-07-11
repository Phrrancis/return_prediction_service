"""
The Money Report: turns predictions + matured outcomes into merchant language.

Sections:
  1. Verified accuracy  - precision/recall/AUC on mature labels ONLY
  2. Euro impact        - return costs among flagged carts (merchant-set cost assumption)
  3. Simulated actions  - expected net profit of the non-returnable coupon at
                          3 operating points, with every assumption explicit
  4. Merchandising cuts - worst SKUs, bracketing hotspots, payment-method split
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

from backend.db import get_conn

# default economic assumptions — every one is merchant-editable in the UI
DEFAULTS = {
    "cost_per_return_eur": 8.50,       # reverse logistics + handling + repackaging
    "coupon_pct": 0.05,                # 5% coupon for accepting non-returnable
    "coupon_adoption": 0.27,           # from the Myntra live experiment
    "return_reduction_if_adopted": 1.0,  # adopted item cannot be returned
}


def generate(merchant_id: str, assumptions: dict | None = None, db_path: str | None = None) -> dict:
    a = {**DEFAULTS, **(assumptions or {})}

    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.score, p.explanations, c.line_items, c.payment,
                   o.returned, o.refund_total, ord.total
            FROM predictions p
            JOIN carts c   ON c.cart_token = p.cart_token AND c.merchant_id = p.merchant_id
            JOIN outcomes o ON o.order_id = c.order_id AND o.merchant_id = c.merchant_id
            JOIN orders ord ON ord.order_id = c.order_id AND ord.merchant_id = c.merchant_id
            WHERE p.merchant_id = ? AND o.matured_at <= datetime('now')
            """,
            (merchant_id,),
        ).fetchall()

    if not rows:
        raise ValueError("No mature, scored carts yet — report would be dishonest.")

    scores = np.array([r["score"] for r in rows])
    y = np.array([r["returned"] for r in rows])

    auc = roc_auc_score(y, scores)
    report = {
        "merchant_id": merchant_id,
        "generated_at": datetime.utcnow().isoformat(),
        "assumptions": a,
        "corpus": {
            "carts_scored_and_mature": int(len(rows)),
            "overall_return_rate": float(y.mean()),
            "verified_auc": float(auc),
        },
        "operating_points": [],
        "merchandising": {},
    }

    # ---- operating points + simulated coupon action -------------------------
    for name, pct in (("conservative", 90), ("balanced", 80), ("aggressive", 65)):
        thr = float(np.percentile(scores, pct))
        flagged = scores >= thr
        n_flag = int(flagged.sum())
        if n_flag == 0:
            continue
        precision = float(y[flagged].mean())
        recall = float(y[flagged].sum() / max(y.sum(), 1))

        # euro impact of flagged carts that actually returned
        returns_cost = float(y[flagged].sum() * a["cost_per_return_eur"])

        # simulated non-returnable coupon on flagged carts:
        # adopters can't return -> save cost_per_return on the would-be returns among adopters
        # coupon cost applies to ALL adopters (kept or returned)
        # coupon applies to the single flagged item (per the Myntra experiment),
        # not the whole cart — avg item price approximated as cart total / items
        flagged_rows = [r for r, f in zip(rows, flagged) if f]
        avg_item_price = float(np.mean([
            (r["total"] or 0.0) / max(len(json.loads(r["line_items"])), 1)
            for r in flagged_rows
        ]))
        adopters = n_flag * a["coupon_adoption"]
        saved = adopters * precision * a["cost_per_return_eur"] * a["return_reduction_if_adopted"]
        coupon_cost = adopters * avg_item_price * a["coupon_pct"]
        net = saved - coupon_cost

        report["operating_points"].append({
            "name": name,
            "score_threshold": round(thr, 3),
            "carts_flagged": n_flag,
            "verified_precision": round(precision, 3),
            "recall_of_all_returns": round(recall, 3),
            "identified_return_cost_eur": round(returns_cost, 2),
            "simulated_coupon_net_profit_eur": round(net, 2),
        })

    # ---- merchandising cuts --------------------------------------------------
    sku_returns, bracketing = {}, {"bracketed": [0, 0], "not": [0, 0]}
    payment_split: dict = {}
    for r in rows:
        items = json.loads(r["line_items"])
        sizes_by_product: dict = {}
        for it in items:
            s = sku_returns.setdefault(it["sku"], [0, 0])
            s[0] += 1
            s[1] += r["returned"]
            sizes_by_product.setdefault(it["product_id"], set()).add(it.get("size"))
        key = "bracketed" if any(len(v) > 1 for v in sizes_by_product.values()) else "not"
        bracketing[key][0] += 1
        bracketing[key][1] += r["returned"]
        p = payment_split.setdefault(r["payment"] or "unknown", [0, 0])
        p[0] += 1
        p[1] += r["returned"]

    worst = sorted(
        ((sku, c[1] / c[0], c[0]) for sku, c in sku_returns.items() if c[0] >= 25),
        key=lambda x: -x[1],
    )[:10]
    report["merchandising"] = {
        "worst_skus": [
            {"sku": s, "return_rate": round(rr, 3), "orders": n} for s, rr, n in worst
        ],
        "bracketing_return_rate": {
            k: round(v[1] / v[0], 3) for k, v in bracketing.items() if v[0] > 0
        },
        "payment_return_rate": {
            k: round(v[1] / v[0], 3) for k, v in payment_split.items() if v[0] > 0
        },
    }
    return report
