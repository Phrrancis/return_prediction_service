"""
ReturnML ingest + admin API.

Endpoints:
  POST /v1/pixel/{merchant_id}      <- Web Pixel events (cart_viewed, checkout_started, ...)
  POST /v1/webhooks/{merchant_id}   <- Shopify webhooks (orders/create, refunds/create, ...)
                                       HMAC-SHA256 verified per Shopify spec
  GET  /v1/health

  Admin (X-API-Key protected) — runs the pipeline on demand instead of only via cron/script:
  POST /v1/train/{merchant_id}      <- train the per-merchant model on matured labels
  POST /v1/score/{merchant_id}      <- shadow-score every cart lacking a prediction
  GET  /v1/report/{merchant_id}     <- the Money Report (verified accuracy + euro impact)

Every ingest payload lands in raw_events first (append-only), then is projected
into carts / orders / outcomes. If a projection ever has a bug, raw_events lets
you rebuild the world.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from backend.auth import verify_api_key
from backend.config import settings
from backend.db import get_conn, init_db, insert_raw_event
from backend.report import generate as generate_report
from backend.score import score_pending
from backend.train import train as train_model

logger = logging.getLogger(__name__)

app = FastAPI(title="ReturnML", version="0.1.0")


@app.on_event("startup")
async def startup() -> None:
    init_db()


def verify_shopify_hmac(body: bytes, hmac_header: str) -> bool:
    digest = hmac.new(settings.shopify_webhook_secret.encode(), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode()
    return hmac.compare_digest(expected, hmac_header or "")


@app.get("/v1/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/pixel/{merchant_id}")
async def pixel(merchant_id: str, request: Request) -> dict[str, bool]:
    payload = await request.json()
    topic = payload.get("event", "unknown")
    with get_conn() as conn:
        insert_raw_event(conn, merchant_id, "pixel", topic, payload)
        if topic in ("checkout_started", "cart_viewed"):
            d = payload.get("data", {})
            conn.execute(
                """INSERT INTO carts (cart_token, merchant_id, line_items, customer_id,
                                      payment, created_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(merchant_id, cart_token) DO UPDATE SET
                     line_items=excluded.line_items,
                     payment=COALESCE(excluded.payment, carts.payment)""",
                (
                    d.get("cart_token"),
                    merchant_id,
                    json.dumps(d.get("line_items", [])),
                    d.get("customer_id"),        # hashed upstream
                    d.get("payment"),
                    d.get("created_at", datetime.utcnow().isoformat()),
                ),
            )
    return {"received": True}


@app.post("/v1/webhooks/{merchant_id}")
async def webhook(
    merchant_id: str,
    request: Request,
    x_shopify_topic: str = Header(default=""),
    x_shopify_hmac_sha256: str = Header(default=""),
) -> dict[str, bool]:
    body = await request.body()
    if not verify_shopify_hmac(body, x_shopify_hmac_sha256):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
    payload = json.loads(body)

    with get_conn() as conn:
        insert_raw_event(conn, merchant_id, "webhook", x_shopify_topic, payload)

        if x_shopify_topic == "orders/create":
            token = payload.get("checkout_token")
            conn.execute(
                """INSERT OR REPLACE INTO orders
                   (order_id, merchant_id, cart_token, customer_id, payment, total, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    str(payload["id"]), merchant_id, token,
                    payload.get("customer_hash"),
                    (payload.get("payment_gateway_names") or [None])[0],
                    payload.get("total_price"),
                    payload.get("created_at", datetime.utcnow().isoformat()),
                ),
            )
            # the identity stitch: cart -> order
            if token:
                conn.execute(
                    "UPDATE carts SET order_id=? WHERE merchant_id=? AND cart_token=?",
                    (str(payload["id"]), merchant_id, token),
                )

        elif x_shopify_topic in ("refunds/create", "returns/create"):
            order_id = str(payload.get("order_id"))
            matured = (datetime.utcnow() + timedelta(
                days=settings.return_window_days + settings.logistics_buffer_days)).isoformat()
            qty = sum(li.get("quantity", 1) for li in payload.get("refund_line_items", [])) or 1
            conn.execute(
                """INSERT OR REPLACE INTO outcomes
                   (order_id, merchant_id, returned, returned_qty, refund_total, reason, matured_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (order_id, merchant_id, 1, qty,
                 payload.get("refund_total", 0.0),
                 payload.get("reason"), matured),
            )
    return {"received": True}


@app.post("/v1/train/{merchant_id}", dependencies=[Depends(verify_api_key)])
def run_train(merchant_id: str) -> dict:
    try:
        metrics, version = train_model(merchant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"version": version, "metrics": metrics}


@app.post("/v1/score/{merchant_id}", dependencies=[Depends(verify_api_key)])
def run_score(merchant_id: str) -> dict:
    try:
        scored = score_pending(merchant_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No trained model for '{merchant_id}'") from exc
    return {"scored": scored}


@app.get("/v1/report/{merchant_id}", dependencies=[Depends(verify_api_key)])
def get_report(merchant_id: str) -> dict:
    try:
        return generate_report(merchant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
