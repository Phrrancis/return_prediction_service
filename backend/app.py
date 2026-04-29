from __future__ import annotations

import logging
import uuid

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from backend.auth import verify_api_key
from backend.database import get_db, init_db, save_prediction
from backend.model import predict_return_risk
from backend.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Return Optimizer", version="2.0.0")


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
async def predict(req: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    try:
        result = predict_return_risk(req)
        save_prediction(
            db,
            prediction_id=str(uuid.uuid4()),
            user_id=req.user_id,
            result=result.model_dump(),
        )
        return result
    except ValueError as exc:
        logger.exception("Prediction failed due to invalid input")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction failure")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
