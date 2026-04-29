from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    cart = Column(JSON, nullable=False)
    score = Column(Float, nullable=False)
    risk = Column(String, nullable=False)
    action = Column(String, nullable=False)
    reasons = Column(JSON, nullable=False)
    features = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():  # noqa: ANN201
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db: Session, prediction_id: str, user_id: str, cart: list, result: dict) -> None:
    record = PredictionRecord(
        id=prediction_id,
        user_id=user_id,
        cart=cart,
        score=result["score"],
        risk=result["risk"],
        action=result["action"],
        reasons=result["reasons"],
        features=result["features"],
    )
    db.add(record)
    db.commit()
