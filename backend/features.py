"""
Feature builder.

ONE pure function used by BOTH training and live scoring. This is deliberate:
training/serving skew (two divergent feature implementations) is the classic
production-ML bug, and a single shared code path is the cheapest insurance.

Feature groups (mirrors the Myntra paper, modernized):
  product-level : price stats, smoothed historical SKU return rates
  cart-level    : size, bracketing (same product, multiple sizes), style dupes,
                  similar-item count, value & discount signals
  user-level    : lifetime return rate, order count, recency (NULL-safe for guests)
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

FEATURE_NAMES = [
    # cart-level
    "cart_size",
    "cart_value",
    "distinct_products",
    "bracketing_flag",        # same product id, >1 size  -> strongest fashion signal
    "same_style_multi_color",
    "n_similar_items",        # near-duplicate titles/categories in cart
    "avg_item_price",
    "max_item_price",
    # product-level (aggregated over cart)
    "mean_sku_return_rate",   # smoothed
    "max_sku_return_rate",
    "mean_category_return_rate",
    # user-level
    "user_order_count",
    "user_return_rate",
    "user_days_since_last",
    "is_guest",
    # context
    "is_cod",                 # cash on delivery
    "is_weekend",
]

SMOOTHING_K = 10  # Bayesian prior weight toward category mean for low-volume SKUs


def smoothed_rate(returns: int, orders: int, prior_rate: float, k: int = SMOOTHING_K) -> float:
    """Beta-binomial style smoothing: low-volume SKUs shrink toward the prior."""
    return (returns + k * prior_rate) / (orders + k) if (orders + k) > 0 else prior_rate


def build_features(cart_row: dict, sku_stats: dict, user_stats: dict, global_return_rate: float = 0.30):
    """
    cart_row  : dict with line_items (list), customer_id, payment, created_at
    sku_stats : {sku: {"orders": int, "returns": int, "category": str}}
                plus {"__category__<cat>": {"orders":..,"returns":..}} entries
    user_stats: {customer_id: {"orders": int, "returns": int, "last_order_days": float}}
    Returns   : (feature_vector list aligned with FEATURE_NAMES, debug dict)
    """
    items = cart_row["line_items"]
    if isinstance(items, str):
        items = json.loads(items)

    prices = [it["price"] for it in items]
    qty_total = sum(it.get("qty", 1) for it in items)
    cart_value = sum(it["price"] * it.get("qty", 1) for it in items)

    # bracketing: same product_id appearing with >1 distinct size
    sizes_by_product: dict = {}
    colors_by_style: dict = {}
    for it in items:
        sizes_by_product.setdefault(it["product_id"], set()).add(it.get("size"))
        style = it.get("style", it["product_id"])
        colors_by_style.setdefault(style, set()).add(it.get("color"))
    bracketing = int(any(len(s) > 1 for s in sizes_by_product.values()))
    multi_color = int(any(len(c) > 1 for c in colors_by_style.values()))

    # similar items: same category appearing 2+ times (cheap proxy; swap for
    # embedding cosine-sim in production)
    cat_counts = Counter(it.get("category", "unknown") for it in items)
    n_similar = sum(c - 1 for c in cat_counts.values() if c > 1)

    # product-level: smoothed SKU return rates
    sku_rates, cat_rates = [], []
    for it in items:
        s = sku_stats.get(it["sku"], {})
        cat = it.get("category", "unknown")
        c = sku_stats.get(f"__category__{cat}", {})
        cat_rate = smoothed_rate(c.get("returns", 0), c.get("orders", 0), global_return_rate)
        sku_rates.append(smoothed_rate(s.get("returns", 0), s.get("orders", 0), cat_rate))
        cat_rates.append(cat_rate)

    # user-level
    cust = cart_row.get("customer_id")
    u = user_stats.get(cust, {}) if cust else {}
    u_orders = u.get("orders", 0)
    u_return_rate = (u.get("returns", 0) / u_orders) if u_orders > 0 else global_return_rate
    u_recency = u.get("last_order_days", 9999.0)

    try:
        dow = datetime.fromisoformat(cart_row["created_at"]).weekday()
    except (ValueError, KeyError, TypeError):
        dow = 0

    vec = [
        qty_total,
        cart_value,
        len({it["product_id"] for it in items}),
        bracketing,
        multi_color,
        n_similar,
        sum(prices) / len(prices) if prices else 0.0,
        max(prices) if prices else 0.0,
        sum(sku_rates) / len(sku_rates) if sku_rates else global_return_rate,
        max(sku_rates) if sku_rates else global_return_rate,
        sum(cat_rates) / len(cat_rates) if cat_rates else global_return_rate,
        u_orders,
        u_return_rate,
        u_recency,
        int(cust is None),
        int((cart_row.get("payment") or "") == "cod"),
        int(dow >= 5),
    ]
    return vec, dict(zip(FEATURE_NAMES, vec))


# ---- human-readable explanation strings from SHAP output --------------------

EXPLAIN_TEMPLATES = {
    "bracketing_flag": "same item in multiple sizes (bracketing)",
    "mean_sku_return_rate": "{:.0%} avg. historical return rate of items in cart",
    "max_sku_return_rate": "one item has {:.0%} historical return rate",
    "user_return_rate": "customer's lifetime return rate is {:.0%}",
    "is_cod": "cash-on-delivery payment",
    "cart_size": "{:.0f} items in cart",
    "n_similar_items": "{:.0f} similar items in cart",
    "same_style_multi_color": "same style in multiple colors",
}


def humanize(feature: str, value: float) -> str:
    tpl = EXPLAIN_TEMPLATES.get(feature)
    if tpl is None:
        return f"{feature.replace('_', ' ')}: {value:.2f}"
    return tpl.format(value) if "{" in tpl else tpl
