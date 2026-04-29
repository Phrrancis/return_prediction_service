from __future__ import annotations

from pydantic import BaseModel


class Product(BaseModel):
    product_id: str
    price: float
    category: str
    size: str


class PredictRequest(BaseModel):
    user_id: str
    cart: list[Product]
