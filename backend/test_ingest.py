from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from backend.config import settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from backend.app import app

    with TestClient(app) as c:
        yield c


def _sign(body: bytes) -> str:
    digest = hmac.new(settings.shopify_webhook_secret.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_pixel_persists_cart(client):
    payload = {
        "event": "cart_viewed",
        "data": {
            "cart_token": "ct_1",
            "line_items": [{"sku": "P1-M", "product_id": "P1", "category": "dresses",
                             "size": "M", "price": 50.0, "qty": 1}],
            "customer_id": "cust1",
            "payment": "card",
            "created_at": "2024-01-01T00:00:00",
        },
    }
    resp = client.post("/v1/pixel/acme", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"received": True}

    from backend.db import get_conn
    with get_conn(settings.db_path) as conn:
        row = conn.execute(
            "SELECT * FROM carts WHERE merchant_id=? AND cart_token=?", ("acme", "ct_1")
        ).fetchone()
    assert row is not None
    assert json.loads(row["line_items"])[0]["sku"] == "P1-M"


def test_webhook_rejects_bad_hmac(client):
    body = json.dumps({"id": 1}).encode()
    resp = client.post(
        "/v1/webhooks/acme",
        content=body,
        headers={"X-Shopify-Topic": "orders/create", "X-Shopify-Hmac-Sha256": "not-valid"},
    )
    assert resp.status_code == 401


def test_webhook_orders_create_stitches_cart_to_order(client):
    from backend.db import get_conn
    with get_conn(settings.db_path) as conn:
        conn.execute(
            """INSERT INTO carts (cart_token, merchant_id, line_items, customer_id, payment, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("ct_2", "acme", "[]", None, "card", "2024-01-01T00:00:00"),
        )

    body = json.dumps({
        "id": 999,
        "checkout_token": "ct_2",
        "total_price": 120.0,
        "payment_gateway_names": ["card"],
        "created_at": "2024-01-01T00:05:00",
    }).encode()
    resp = client.post(
        "/v1/webhooks/acme",
        content=body,
        headers={"X-Shopify-Topic": "orders/create", "X-Shopify-Hmac-Sha256": _sign(body)},
    )
    assert resp.status_code == 200

    with get_conn(settings.db_path) as conn:
        cart = conn.execute(
            "SELECT order_id FROM carts WHERE merchant_id=? AND cart_token=?", ("acme", "ct_2")
        ).fetchone()
        order = conn.execute(
            "SELECT * FROM orders WHERE merchant_id=? AND order_id=?", ("acme", "999")
        ).fetchone()
    assert cart["order_id"] == "999"
    assert order["total"] == 120.0


def test_webhook_refund_creates_matured_outcome(client):
    body = json.dumps({
        "order_id": 555,
        "refund_total": 42.0,
        "reason": "size_fit",
        "refund_line_items": [{"quantity": 2}],
    }).encode()
    resp = client.post(
        "/v1/webhooks/acme",
        content=body,
        headers={"X-Shopify-Topic": "refunds/create", "X-Shopify-Hmac-Sha256": _sign(body)},
    )
    assert resp.status_code == 200

    from backend.db import get_conn
    with get_conn(settings.db_path) as conn:
        outcome = conn.execute(
            "SELECT * FROM outcomes WHERE merchant_id=? AND order_id=?", ("acme", "555")
        ).fetchone()
    assert outcome["returned"] == 1
    assert outcome["returned_qty"] == 2
    assert outcome["matured_at"] > "2024"  # future ISO timestamp, well past this fixed literal
