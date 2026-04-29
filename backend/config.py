from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Risk score thresholds
    high_risk_threshold: float = 0.6
    medium_risk_threshold: float = 0.3

    # Scoring penalties
    high_cart_size_threshold: int = 3
    similar_items_threshold: int = 1
    low_price_threshold: float = 30.0
    high_cart_size_penalty: float = 0.4
    similar_items_penalty: float = 0.4
    low_price_penalty: float = 0.1

    # Auth
    api_key: str = "dev-key"

    # Database
    database_url: str = "sqlite:///./predictions.db"

    # ML model path
    ml_model_path: str = "model_artifacts/model.pkl"


settings = Settings()
