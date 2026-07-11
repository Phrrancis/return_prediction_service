from __future__ import annotations

from backend.features import FEATURE_NAMES, build_features, humanize, smoothed_rate


def _cart(line_items, customer_id=None, payment=None, created_at="2024-01-01T00:00:00"):
    return {
        "line_items": line_items,
        "customer_id": customer_id,
        "payment": payment,
        "created_at": created_at,
    }


def test_smoothed_rate_shrinks_low_volume_toward_prior():
    # zero orders -> falls back entirely to the prior
    assert smoothed_rate(0, 0, prior_rate=0.3) == 0.3
    # low volume shrinks toward the prior rather than reporting the raw rate
    low_volume = smoothed_rate(returns=1, orders=1, prior_rate=0.3)
    assert 0.3 < low_volume < 1.0
    # high volume converges toward the observed rate
    high_volume = smoothed_rate(returns=800, orders=1000, prior_rate=0.3)
    assert abs(high_volume - 0.8) < 0.05


def test_build_features_detects_bracketing():
    items = [
        {"sku": "P1-M", "product_id": "P1", "category": "dresses", "size": "M", "price": 50.0, "qty": 1},
        {"sku": "P1-L", "product_id": "P1", "category": "dresses", "size": "L", "price": 50.0, "qty": 1},
    ]
    _, features = build_features(_cart(items), sku_stats={}, user_stats={})
    assert features["bracketing_flag"] == 1
    assert features["cart_size"] == 2
    assert features["distinct_products"] == 1


def test_build_features_no_bracketing_for_distinct_products():
    items = [
        {"sku": "P1-M", "product_id": "P1", "category": "dresses", "size": "M", "price": 50.0, "qty": 1},
        {"sku": "P2-M", "product_id": "P2", "category": "jeans", "size": "M", "price": 60.0, "qty": 1},
    ]
    _, features = build_features(_cart(items), sku_stats={}, user_stats={})
    assert features["bracketing_flag"] == 0
    assert features["distinct_products"] == 2


def test_build_features_uses_sku_and_user_stats():
    items = [{"sku": "P1-M", "product_id": "P1", "category": "dresses", "size": "M", "price": 50.0, "qty": 1}]
    sku_stats = {
        "P1-M": {"orders": 100, "returns": 80},
        "__category__dresses": {"orders": 1000, "returns": 300},
    }
    user_stats = {"cust1": {"orders": 10, "returns": 6, "last_order_days": 5.0}}
    vec, features = build_features(
        _cart(items, customer_id="cust1", payment="cod"),
        sku_stats, user_stats, global_return_rate=0.3,
    )
    assert len(vec) == len(FEATURE_NAMES)
    assert features["mean_sku_return_rate"] > 0.5  # high historical return rate carries through
    assert features["user_return_rate"] == 0.6
    assert features["is_cod"] == 1
    assert features["is_guest"] == 0


def test_build_features_handles_guest_checkout():
    items = [{"sku": "P1-M", "product_id": "P1", "category": "dresses", "size": "M", "price": 50.0, "qty": 1}]
    _, features = build_features(_cart(items, customer_id=None), sku_stats={}, user_stats={})
    assert features["is_guest"] == 1
    assert features["user_return_rate"] == 0.3  # falls back to global_return_rate default


def test_humanize_known_and_unknown_features():
    assert "bracketing" in humanize("bracketing_flag", 1)
    assert humanize("mean_sku_return_rate", 0.42) == "42% avg. historical return rate of items in cart"
    assert "some made up feature" in humanize("some_made_up_feature", 1.2345)
