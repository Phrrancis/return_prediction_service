from __future__ import annotations

from backend.features import FeatureSet, extract_features
from backend.schemas import PredictRequest

HIGH_CART_SIZE_THRESHOLD = 3
SIMILAR_ITEMS_THRESHOLD = 1
LOW_PRICE_THRESHOLD = 30
HIGH_RISK_SCORE_THRESHOLD = 0.7
MEDIUM_RISK_SCORE_THRESHOLD = 0.4


def _score_features(features: FeatureSet) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if features["cart_size"] > HIGH_CART_SIZE_THRESHOLD:
        score += 0.3
        reasons.append(f"cart_size > {HIGH_CART_SIZE_THRESHOLD}")

    if features["similar_items"] > SIMILAR_ITEMS_THRESHOLD:
        score += 0.3
        reasons.append(f"similar_items > {SIMILAR_ITEMS_THRESHOLD}")

    if features["avg_price"] < LOW_PRICE_THRESHOLD:
        score -= 0.1
        reasons.append(f"avg_price < {LOW_PRICE_THRESHOLD}")

    return min(max(score, 0.0), 1.0), reasons


def _risk_from_score(score: float) -> tuple[str, str]:
    if score > HIGH_RISK_SCORE_THRESHOLD:
        return "high", "add_shipping_fee"
    if score > MEDIUM_RISK_SCORE_THRESHOLD:
        return "medium", "show_warning"
    return "low", "none"


def predict_return_risk(req: PredictRequest) -> dict[str, object]:
    features = extract_features(req.cart)
    score, reasons = _score_features(features)
    risk, action = _risk_from_score(score)

    return {
        "score": score,
        "risk": risk,
        "action": action,
        "reasons": reasons,
        "features": features,
    }