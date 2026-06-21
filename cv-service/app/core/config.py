from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
from typing import List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def ensure_rediss_ssl_cert_reqs(redis_url: str) -> str:
    if not redis_url.startswith("rediss://"):
        return redis_url

    parts = urlsplit(redis_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    if "ssl_cert_reqs" in query:
        return redis_url

    query["ssl_cert_reqs"] = "CERT_REQUIRED"
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


class Settings(BaseSettings):
    # Server
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 1
    api_docs_enabled: bool = True

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

    # AI provider configuration (remote_api | gemini)
    ai_provider: str = "gemini"
    # Gemini API configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.9
    gemini_top_p: float = 0.95
    gemini_min_dish_suggestions: int = 3
    gemini_max_dish_suggestions: int = 5

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
    user_profile_enabled: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    # Nutrition cache
    nutrition_cache_ttl: int = 86400          # seconds — Redis TTL for nutrition entries
    usda_name_match_threshold: float = 0.35   # min SequenceMatcher ratio OR word overlap to accept USDA result
    usda_max_retries: int = 3                 # retries after HTTP 429 from FoodData Central
    usda_retry_backoff_seconds: float = 2.0   # base delay; doubled each retry (exponential backoff)
    nutrition_enrichment_enabled: bool = True # normalize labels + attach nutrition_breakdown after AI inference

    # Security
    api_secret_key: str = ""                  # backend → AI service bearer token
    auth_enabled: bool = True
    admin_api_key: str = ""                   # required for /cv/admin/* endpoints
    allowed_origins: List[str] = ["http://localhost"]

    # Logging
    log_level: str = "INFO"



    @model_validator(mode="after")
    def require_secrets_in_production(self) -> "Settings":
        if (
            self.redis_url == "redis://localhost:6379/0"
            and self.upstash_redis_rest_url.startswith(("redis://", "rediss://"))
        ):
            self.redis_url = self.upstash_redis_rest_url

        if self.app_env == "production" and not self.auth_enabled:
            raise ValueError("AUTH_ENABLED cannot be false in production")

        if self.app_env == "production":
            required = {
                "USDA_API_KEY": self.usda_api_key,
                "API_SECRET_KEY": self.api_secret_key,
            }
            if self.ai_provider == "gemini":
                required["GEMINI_API_KEY"] = self.gemini_api_key
            elif self.ai_provider == "remote_api":
                required["AI_API_KEY"] = self.ai_api_key
                required["AI_API_BASE_URL"] = self.ai_api_base_url
            missing = [k for k, v in required.items() if not v]
            if missing:
                raise ValueError(f"Required env vars not set: {', '.join(missing)}")
        return self

    @field_validator("allowed_mime_types", "allowed_origins", mode="before")
    @classmethod
    def split_comma(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @field_validator("redis_url", "upstash_redis_rest_url", mode="after")
    @classmethod
    def normalize_redis_url(cls, v: str) -> str:
        return ensure_rediss_ssl_cert_reqs(v) if isinstance(v, str) else v

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
