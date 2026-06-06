from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
from typing import List


class Settings(BaseSettings):
    # Server
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 1

    # Model weights / device
    food_detection_weights: str = "weights/yolov8_food.pt"
    food_classify_weights: str = "weights/efficientnet_food.pt"
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "cpu"

    # Image validation
    max_image_size_mb: int = 10
    allowed_mime_types: List[str] = ["image/jpeg", "image/png", "image/webp"]

    # Nutrition API
    usda_api_key: str = ""
    usda_base_url: str = "https://api.nal.usda.gov/fdc/v1"

    # PostgREST + vector semantic search
    postgrest_url: str = ""
    postgrest_api_key: str = ""
    postgrest_service_jwt: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64
    vector_similarity_threshold: float = 0.82

    # CLIP zero-shot classifier (Option 2)
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # Auth + meal history (Option 3)
    meal_history_enabled: bool = True
    meal_history_similarity_threshold: float = 0.3

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Nutrition cache
    nutrition_cache_ttl: int = 86400          # seconds — Redis TTL for nutrition entries
    usda_name_match_threshold: float = 0.35   # min SequenceMatcher ratio OR word overlap to accept USDA result

    # Security
    api_secret_key: str = ""
    admin_api_key: str = ""                   # required for /cv/admin/* endpoints
    allowed_origins: List[str] = ["http://localhost"]

    # Logging
    log_level: str = "INFO"

    # Pipeline stage selection (swap models via config)
    pipeline_detector: str = "yolov8"
    pipeline_classifier: str = "efficientnet_b4"
    pipeline_depth: str = "depth_anything_v2"
    pipeline_postprocessor: str = "default"

    @property
    def pipeline_config(self) -> dict:
        """Config dict consumed by ``PipelineFactory.build()``."""
        return {
            "detector": self.pipeline_detector,
            "classifier": self.pipeline_classifier,
            "depth": self.pipeline_depth,
            "postprocessor": self.pipeline_postprocessor,
        }

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


settings = Settings()
