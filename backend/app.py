from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from backend.model import predict_return_risk
from backend.schemas import PredictRequest

logger = logging.getLogger(__name__)

app = FastAPI(title="Return Optimizer", version="1.0.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest) -> dict[str, object]:
    try:
        return predict_return_risk(req)
    except ValueError as exc:
        logger.exception("Prediction failed due to invalid input")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction failure")
        raise HTTPException(status_code=500, detail="Internal server error") from exc