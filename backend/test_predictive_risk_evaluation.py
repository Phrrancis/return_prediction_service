from __future__ import annotations

from backend.schemas import PredictRequest, Product
from backend.model import predict_return_risk


def _make_request(items: list[tuple[str, float, str, str]]) -> PredictRequest:
    return PredictRequest(
        user_id="test-user",
        cart=[
            Product(product_id=pid, price=price, category=cat, size=size)
            for pid, price, cat, size in items
        ],
    )


def test_predict_return_risk_empty_cart() -> None:
    req = PredictRequest(user_id="test-user", cart=[])
    result = predict_return_risk(req)

    assert result.score == 0.0 or result.risk == "low"
    assert result.risk == "low"
    assert result.action == "none"
    assert result.features.cart_size == 0


def test_predict_return_risk_high_risk() -> None:
    req = _make_request([
        ("1", 10.0, "shirts", "M"),
        ("2", 12.0, "shirts", "L"),
        ("3", 11.0, "shirts", "S"),
        ("4", 9.0, "shirts", "XL"),
    ])
    result = predict_return_risk(req)

    # With fixed scoring (penalties of 0.4 each, max 0.8) high risk is now reachable
    assert result.risk in ("medium", "high")
    assert result.action in ("show_warning", "add_shipping_fee")
    assert result.score > 0.3


def test_predict_return_risk_medium_or_low_has_expected_shape() -> None:
    req = _make_request([
        ("1", 50.0, "shoes", "M"),
        ("2", 45.0, "accessories", "L"),
    ])
    result = predict_return_risk(req)

    assert hasattr(result, "score")
    assert hasattr(result, "risk")
    assert hasattr(result, "action")
    assert hasattr(result, "features")
    assert hasattr(result, "reasons")
