from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
from typing import List


class Settings(BaseSettings):
    # Server
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 1

    # Runtime device configuration (kept for metadata/logging)
    device: str = "cpu"

    # Image validation
    max_image_size_mb: int = 25
    allowed_mime_types: List[str] = ["image/jpeg", "image/png", "image/webp"]

    # Nutrition API
    usda_api_key: str = ""
    usda_base_url: str = "https://api.nal.usda.gov/fdc/v1"

    # AI inference API (remote model)
    ai_api_base_url: str = ""
    ai_api_key: str = ""
    ai_api_timeout_seconds: float = 30.0
    ai_api_poll_interval_seconds: float = 1.5

    # AI provider configuration (remote_api | gemini | mock)
    ai_provider: str = "mock"
    # Gemini API configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # PostgREST + vector semantic search
    postgrest_url: str = ""
    postgrest_api_key: str = ""
    postgrest_service_jwt: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64
    vector_similarity_threshold: float = 0.82



    # Auth + meal history (Option 3)
    meal_history_enabled: bool = True
    meal_history_similarity_threshold: float = 0.3

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Nutrition cache
    nutrition_cache_ttl: int = 86400          # seconds — Redis TTL for nutrition entries
    usda_name_match_threshold: float = 0.35   # min SequenceMatcher ratio OR word overlap to accept USDA result

    # Security
    api_secret_key: str = ""                  # backend → AI service bearer token
    admin_api_key: str = ""                   # required for /cv/admin/* endpoints
    allowed_origins: List[str] = ["http://localhost"]

    # Logging
    log_level: str = "INFO"



    @model_validator(mode="after")
    def require_secrets_in_production(self) -> "Settings":
        if self.app_env == "production":
            missing = [
                k
                for k, v in {
                    "USDA_API_KEY": self.usda_api_key,
                    "API_SECRET_KEY": self.api_secret_key,
                }.items()
                if not v
            ]
            if missing:
                raise ValueError(f"Required env vars not set: {', '.join(missing)}")
        return self

    @field_validator("allowed_mime_types", "allowed_origins", mode="before")
    @classmethod
    def split_comma(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
