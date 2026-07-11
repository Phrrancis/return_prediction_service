# ReturnML — Shadow-Mode Return Prediction Service

Score every cart's return risk silently, verify predictions against real outcomes,
and show merchants the money — before asking them to change anything at checkout.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_demo.py            # full pipeline on a synthetic merchant
uvicorn backend.app:app --reload   # the ingest + admin API
```

## Run tests

```bash
PYTHONPATH=. pytest backend/ -v
```

## What the demo does

1. **Generates** 18 months of synthetic fashion-merchant history (20K carts) with
   realistic causal structure: bracketing (same item, 2 sizes), bad-fit SKUs,
   serial returners, COD effect.
2. **Trains** a per-merchant LightGBM model on a **temporal split** (no leakage:
   SKU return-rate features computed from the training window only).
3. **Shadow-scores** every cart with SHAP explanations — no customer impact.
4. **Matures labels**: an outcome only counts after return window + buffer.
5. **Generates the Money Report**: verified precision/recall, euros of identified
   return cost, and simulated per-item coupon profit at 3 operating points.

## Architecture

```
Web Pixel ──cart events──▶ POST /v1/pixel/{merchant_id} ─┐
Shopify  ──orders/create─▶ POST /v1/webhooks/{merchant_id} (HMAC) ─▶ raw_events (append-only)
         ──refunds/create▶                                          │
                                                       carts ⨝ orders ⨝ outcomes
                                                                   │
                                POST /v1/train/{merchant_id}  → backend/train.py (temporal split, LightGBM)
                                                                   │
                                POST /v1/score/{merchant_id}  → backend/score.py (shadow scoring + SHAP)
                                                                   │
                                GET  /v1/report/{merchant_id} → backend/report.py (the Money Report)
```

Five tables: `raw_events`, `carts`, `orders`, `predictions` (append-only),
`outcomes` (label-maturity enforced). SQLite for the demo; schema is
Postgres-ready. See `backend/db.py`.

The `/v1/train`, `/v1/score`, and `/v1/report` endpoints are protected by
`X-API-Key` (see `backend/auth.py`) — they let you run the pipeline on a live
service instead of only via `run_demo.py`. Shopify webhooks authenticate via
HMAC-SHA256 instead, per the Shopify spec.

## Honest caveats (read before demoing to a merchant)

- **Demo AUC is optimistic**: the report scores carts the model trained on.
  The temporal-validation AUC (what a live pilot would see on unseen carts) is
  what matters — the training step prints it. In production, live scoring is
  always out-of-sample by construction.
- **The coupon simulation is assumption-driven** (adoption 27%, cost/return
  €8.50, coupon 5% per item). Every number is in `backend/report.py`'s
  `DEFAULTS` and should be merchant-editable in the UI — that transparency is
  the trust mechanism.
- **Synthetic data is a smoke test**, not a benchmark. Real merchant backfills
  will have messier categories, guest checkout rates, and label noise.

## Production TODO

- [ ] Shopify app scaffold (`shopify app init`), OAuth scopes, Web Pixel extension
- [ ] GraphQL bulk backfill (orders + refunds, 12–24 months)
- [ ] Postgres migration + per-merchant row isolation
- [ ] Nightly maturation & retraining jobs (cron/worker) instead of on-demand admin calls
- [ ] Report UI in embedded admin (Polaris) + PDF export
- [ ] Multimodal product embeddings for `n_similar_items` (replaces category proxy)
- [ ] Observability: alert on "no events in 6h" per merchant
- [ ] DPA template; hash customer IDs at ingest (already stubbed)
