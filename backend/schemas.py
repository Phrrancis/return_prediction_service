from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


ALLOWED_CATEGORIES = {
    "shirts", "pants", "shoes", "boots", "jackets", "accessories",
    "socks", "hats", "bags", "watches", "electronics", "other",
}


class Product(BaseModel):
    product_id: str
    price: float
    category: str
    size: str

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price must be greater than 0")
        return v

    @field_validator("category")
    @classmethod
    def category_must_be_valid(cls, v: str) -> str:
        normalised = v.strip().lower()
        if normalised not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"category '{v}' is not allowed. Must be one of: {sorted(ALLOWED_CATEGORIES)}"
            )
        return normalised

    @field_validator("product_id")
    @classmethod
    def product_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("product_id must not be blank")
        return v.strip()


class PredictRequest(BaseModel):
    user_id: str
    cart: list[Product]

    @model_validator(mode="after")
    def no_duplicate_product_ids(self) -> PredictRequest:
        ids = [p.product_id for p in self.cart]
        if len(ids) != len(set(ids)):
            raise ValueError("cart contains duplicate product_id values")
        return self

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id must not be blank")
        return v.strip()


class FeatureSummary(BaseModel):
    cart_size: int
    avg_price: float
    similar_items: int
    price_variance: float
    category_diversity: float


class PredictResponse(BaseModel):
    score: float
    risk: str
    action: str
    reasons: list[str]
    features: FeatureSummary
