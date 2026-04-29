from __future__ import annotations

import logging
import pickle
from pathlib import Path

from backend.config import settings
from backend.features import FeatureSet, extract_features
from backend.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ML model (optional) — loaded once at import time if the artifact exists
# ---------------------------------------------------------------------------

_ml_model = None

def _load_ml_model() -> object | None:
    path = Path(settings.ml_model_path)
    if path.exists():
        with path.open("rb") as f:
            logger.info("Loaded ML model from %s", path)
            return pickle.load(f)
    logger.info("No ML model artifact found at %s — using rule-based scorer", path)
    return None


_ml_model = _load_ml_model()


# ---------------------------------------------------------------------------
# Rule-based fallback scorer
# ---------------------------------------------------------------------------

def _score_features(features: FeatureSet) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if features["cart_size"] > settings.high_cart_size_threshold:
        score += settings.high_cart_size_penalty
        reasons.append(f"cart_size > {settings.high_cart_size_threshold}")

    if features["similar_items"] > settings.similar_items_threshold:
        score += settings.similar_items_penalty
        reasons.append(f"similar_items > {settings.similar_items_threshold}")

    if features["avg_price"] < settings.low_price_threshold:
        score -= settings.low_price_penalty
        reasons.append(f"avg_price < {settings.low_price_threshold}")

    return min(max(score, 0.0), 1.0), reasons


def _risk_from_score(score: float) -> tuple[str, str]:
    if score >= settings.high_risk_threshold:
        return "high", "add_shipping_fee"
    if score >= settings.medium_risk_threshold:
        return "medium", "show_warning"
    return "low", "none"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_return_risk(req: PredictRequest) -> PredictResponse:
    features = extract_features(req.cart)

    if _ml_model is not None:
        feature_vector = [[
            features["cart_size"],
            features["avg_price"],
            features["similar_items"],
            features["price_variance"],
            features["category_diversity"],
        ]]
        score = float(_ml_model.predict_proba(feature_vector)[0][1])
        reasons = ["ml_model_prediction"]
    else:
        score, reasons = _score_features(features)

    risk, action = _risk_from_score(score)

    return PredictResponse(
        score=score,
        risk=risk,
        action=action,
        reasons=reasons,
        features=features,
    )
