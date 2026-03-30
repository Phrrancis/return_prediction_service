from __future__ import annotations

from backend.model import predict_return_risk


class DummyProduct:
    def __init__(self, product_id: str, price: float, category: str, size: str) -> None:
        self.product_id = product_id
        self.price = price
        self.category = category
        self.size = size


class DummyRequest:
    def __init__(self, cart: list[DummyProduct]) -> None:
        self.cart = cart


def test_predict_return_risk_empty_cart() -> None:
    result = predict_return_risk(DummyRequest(cart=[]))

    assert result["score"] == 0.0
    assert result["risk"] == "low"
    assert result["action"] == "none"
    assert result["features"]["cart_size"] == 0


def test_predict_return_risk_high_risk() -> None:
    cart = [
        DummyProduct("1", 10.0, "shirts", "M"),
        DummyProduct("2", 12.0, "shirts", "L"),
        DummyProduct("3", 11.0, "shirts", "S"),
        DummyProduct("4", 9.0, "shirts", "XL"),
    ]

    result = predict_return_risk(DummyRequest(cart=cart))

    assert result["risk"] == "high"
    assert result["action"] == "add_shipping_fee"
    assert result["score"] > 0.7


def test_predict_return_risk_medium_or_low_has_expected_shape() -> None:
    cart = [
        DummyProduct("1", 50.0, "shoes", "M"),
        DummyProduct("2", 45.0, "accessories", "L"),
    ]

    result = predict_return_risk(DummyRequest(cart=cart))

    assert "score" in result
    assert "risk" in result
    assert "action" in result
    assert "features" in result
    assert "reasons" in result
