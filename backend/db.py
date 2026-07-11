"""
ReturnML database layer.

Runs on SQLite for zero-setup portability; the schema is written to be
Postgres-compatible (swap connect() and the few AUTOINCREMENT usages).
Five tables carry the entire product:

  raw_events   - append-only log of every pixel event & webhook (rebuild anything from this)
  carts        - one row per cart/checkout token; the identity stitch lives here (cart -> order)
  orders       - order facts needed for features & joining
  predictions  - append-only model outputs (score + SHAP explanations), the audit trail
  outcomes     - matured ground-truth labels (returned or kept), only valid past maturity date
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend.config import settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_db_path(db_path: str | Path | None) -> Path:
    path = Path(db_path or settings.db_path)
    return path if path.is_absolute() else REPO_ROOT / path


DB_PATH = _resolve_db_path(None)

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant_id TEXT NOT NULL,
    source      TEXT NOT NULL,           -- 'pixel' | 'webhook'
    topic       TEXT NOT NULL,           -- e.g. 'checkout_started', 'orders/create'
    payload     TEXT NOT NULL,           -- raw JSON, never mutated
    received_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS carts (
    cart_token  TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    line_items  TEXT NOT NULL,           -- JSON [{sku, product_id, title, category, size, qty, price}]
    customer_id TEXT,                    -- hashed; NULL for guests
    payment     TEXT,
    created_at  TEXT NOT NULL,
    order_id    TEXT,                    -- filled when orders/create arrives (identity stitch)
    PRIMARY KEY (merchant_id, cart_token)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    cart_token  TEXT,
    customer_id TEXT,
    payment     TEXT,
    total       REAL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (merchant_id, order_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant_id   TEXT NOT NULL,
    cart_token    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    score         REAL NOT NULL,         -- P(return)
    explanations  TEXT,                  -- JSON top SHAP features, human-readable
    scored_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS outcomes (
    order_id     TEXT NOT NULL,
    merchant_id  TEXT NOT NULL,
    returned     INTEGER NOT NULL,       -- 1 if any line item returned
    returned_qty INTEGER DEFAULT 0,
    refund_total REAL DEFAULT 0,
    reason       TEXT,
    matured_at   TEXT NOT NULL,          -- label only trustworthy from this date
    PRIMARY KEY (merchant_id, order_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_cart ON predictions (merchant_id, cart_token);
CREATE INDEX IF NOT EXISTS idx_cart_order ON carts (merchant_id, order_id);
"""


def init_db(db_path: str | Path | None = None) -> None:
    path = _resolve_db_path(db_path)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


@contextmanager
def get_conn(db_path: str | Path | None = None):
    conn = sqlite3.connect(_resolve_db_path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_raw_event(conn: sqlite3.Connection, merchant_id: str, source: str, topic: str, payload: dict) -> None:
    conn.execute(
        "INSERT INTO raw_events (merchant_id, source, topic, payload) VALUES (?,?,?,?)",
        (merchant_id, source, topic, json.dumps(payload)),
    )
