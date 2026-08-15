"""DataPilot — Configuration module."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Google AI
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    # Model names
    model_high: str = Field(default="gemini-3.5-flash-lite", alias="MODEL_HIGH")
    model_medium: str = Field(default="gemini-3.1-flash-lite", alias="MODEL_MEDIUM")
    model_low: str = Field(default="gemma-4-31b-it", alias="MODEL_LOW")

    # Upstash Redis
    upstash_redis_rest_url: Optional[str] = Field(default=None, alias="UPSTASH_REDIS_REST_URL")
    upstash_redis_rest_token: Optional[str] = Field(default=None, alias="UPSTASH_REDIS_REST_TOKEN")

    # Cache
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    semantic_cache_threshold: float = Field(default=0.92, alias="SEMANTIC_CACHE_THRESHOLD")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500",
        alias="CORS_ORIGINS",
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Paths
    data_dir: str = Field(default="data", alias="DATA_DIR")
    upload_dir: str = Field(default="data/uploads", alias="UPLOAD_DIR")

    model_config = {
        "env_file": str(Path(__file__).parent.parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def redis_available(self) -> bool:
        return bool(self.upstash_redis_rest_url and self.upstash_redis_rest_token)

    @property
    def data_path(self) -> Path:
        return Path(__file__).parent.parent / self.data_dir

    @property
    def upload_path(self) -> Path:
        path = Path(__file__).parent.parent / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
