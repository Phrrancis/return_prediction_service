from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Auth for admin endpoints (train/score/report). Shopify webhooks use HMAC instead.
    api_key: str = "dev-key"

    # Shopify webhook HMAC verification
    shopify_webhook_secret: str = "dev-secret"

    # Label maturation: an outcome is only trustworthy after the return window
    # plus a logistics buffer has elapsed since order creation.
    return_window_days: int = 30
    logistics_buffer_days: int = 14

    # SQLite database file (5-table event-sourced schema, see backend/db.py)
    db_path: str = "returnml.db"


settings = Settings()
