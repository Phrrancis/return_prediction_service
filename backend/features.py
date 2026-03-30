from __future__ import annotations

from typing import Any, TypedDict


class FeatureSet(TypedDict):
    cart_size: int
    avg_price: float
    similar_items: int
    price_variance: float
    category_diversity: float


def extract_features(cart: list[Any]) -> FeatureSet:
    if not cart:
        return {
            "cart_size": 0,
            "avg_price": 0.0,
            "similar_items": 0,
            "price_variance": 0.0,
            "category_diversity": 0.0,
        }

    cart_size = len(cart)
    prices = [p.price for p in cart]
    avg_price = sum(prices) / cart_size

    categories = [p.category.strip().lower() for p in cart]
    unique_categories = len(set(categories))
    similar_items = cart_size - unique_categories

    price_variance = sum((price - avg_price) ** 2 for price in prices) / cart_size
    category_diversity = unique_categories / cart_size

    return {
        "cart_size": cart_size,
        "avg_price": avg_price,
        "similar_items": similar_items,
        "price_variance": price_variance,
        "category_diversity": category_diversity,
    }