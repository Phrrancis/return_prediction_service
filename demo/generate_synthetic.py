"""
Synthetic fashion merchant: 18 months of carts, orders and (matured) outcomes.

The generator encodes the real-world causal structure the model should find:
  - bracketing (same product, 2 sizes) massively raises return probability
  - high historical-return SKUs (bad fit/quality) keep getting returned
  - serial returners exist; COD raises returns; big carts return more
This gives the demo teeth: if the model works, it must rediscover these.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta

from backend.db import get_conn, init_db

random.seed(42)

CATEGORIES = ["dresses", "jeans", "tops", "outerwear", "shoes", "accessories"]
CAT_BASE_RETURN = {"dresses": 0.42, "jeans": 0.38, "tops": 0.25,
                    "outerwear": 0.28, "shoes": 0.35, "accessories": 0.10}
SIZES = {"dresses": ["XS", "S", "M", "L", "XL"], "jeans": ["W28", "W30", "W32", "W34"],
         "tops": ["XS", "S", "M", "L", "XL"], "outerwear": ["S", "M", "L", "XL"],
         "shoes": ["37", "38", "39", "40", "41"], "accessories": ["one"]}
COLORS = ["black", "white", "red", "navy", "sand", "cream"]


def make_catalog(n_products: int = 400) -> list[dict]:
    catalog = []
    for i in range(n_products):
        cat = random.choice(CATEGORIES)
        # each product gets an intrinsic "fit problem" multiplier
        fit_problem = random.choices([0.6, 1.0, 1.6], weights=[0.2, 0.6, 0.2])[0]
        catalog.append({
            "product_id": f"P{i:04d}",
            "category": cat,
            "style": f"S{i:04d}",
            "price": round(random.uniform(25, 180), 2),
            "fit_problem": fit_problem,
        })
    return catalog


def make_customers(n: int = 3000) -> list[dict]:
    customers = []
    for i in range(n):
        persona = random.choices(["keeper", "normal", "serial"], weights=[0.3, 0.55, 0.15])[0]
        mult = {"keeper": 0.4, "normal": 1.0, "serial": 2.2}[persona]
        customers.append({
            "id": hashlib.sha256(f"cust{i}".encode()).hexdigest()[:16],
            "return_mult": mult,
        })
    return customers


def generate(merchant_id: str = "modanova", n_carts: int = 20000, db_path: str | None = None) -> None:
    init_db(db_path)
    catalog = make_catalog()
    customers = make_customers()
    start = datetime.utcnow() - timedelta(days=550)

    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM carts WHERE merchant_id=?", (merchant_id,))
        conn.execute("DELETE FROM orders WHERE merchant_id=?", (merchant_id,))
        conn.execute("DELETE FROM outcomes WHERE merchant_id=?", (merchant_id,))
        conn.execute("DELETE FROM predictions WHERE merchant_id=?", (merchant_id,))

        for i in range(n_carts):
            created = start + timedelta(minutes=random.uniform(0, 550 * 24 * 60))
            cust = random.choice(customers) if random.random() > 0.25 else None
            payment = random.choices(["card", "cod", "invoice"], weights=[0.6, 0.25, 0.15])[0]

            # build cart
            n_items = random.choices([1, 2, 3, 4, 5, 6], weights=[35, 28, 17, 10, 6, 4])[0]
            base_products = random.sample(catalog, k=min(n_items, len(catalog)))
            items, bracketed = [], False
            for p in base_products:
                size = random.choice(SIZES[p["category"]])
                items.append({"sku": f"{p['product_id']}-{size}", "product_id": p["product_id"],
                              "style": p["style"], "category": p["category"], "size": size,
                              "color": random.choice(COLORS), "qty": 1, "price": p["price"]})
                # bracketing: sometimes add same product in a second size
                if random.random() < 0.12 and len(SIZES[p["category"]]) > 1:
                    other = random.choice([s for s in SIZES[p["category"]] if s != size])
                    items.append({**items[-1], "sku": f"{p['product_id']}-{other}", "size": other})
                    bracketed = True

            # ---- ground-truth return probability (the causal model) ----
            cat_component = sum(CAT_BASE_RETURN[it["category"]] for it in items) / len(items)
            fit_component = sum(
                next(p["fit_problem"] for p in catalog if p["product_id"] == it["product_id"])
                for it in items) / len(items)
            p_return = cat_component * fit_component
            if bracketed:
                p_return *= 1.9
            if payment == "cod":
                p_return *= 1.35
            if len(items) >= 5:
                p_return *= 1.3
            if cust:
                p_return *= cust["return_mult"]
            p_return = min(p_return, 0.95)

            returned = int(random.random() < p_return)
            cart_token = f"ct_{i:06d}"
            order_id = f"ord_{i:06d}"

            conn.execute(
                """INSERT INTO carts (cart_token, merchant_id, line_items, customer_id,
                                      payment, created_at, order_id)
                   VALUES (?,?,?,?,?,?,?)""",
                (cart_token, merchant_id, json.dumps(items),
                 cust["id"] if cust else None, payment, created.isoformat(), order_id))
            total = sum(it["price"] for it in items)
            conn.execute(
                """INSERT INTO orders (order_id, merchant_id, cart_token, customer_id,
                                       payment, total, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (order_id, merchant_id, cart_token,
                 cust["id"] if cust else None, payment, total, created.isoformat()))
            # matured label (demo compresses time: all history is past maturity)
            matured = created + timedelta(days=44)
            refund = round(total * random.uniform(0.4, 1.0), 2) if returned else 0.0
            conn.execute(
                """INSERT INTO outcomes (order_id, merchant_id, returned, returned_qty,
                                         refund_total, reason, matured_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (order_id, merchant_id, returned, int(returned), refund,
                 random.choice(["size_fit", "not_as_pictured", "changed_mind"]) if returned else None,
                 matured.isoformat()))

    print(f"Generated {n_carts} carts for merchant '{merchant_id}'")


if __name__ == "__main__":
    generate()
