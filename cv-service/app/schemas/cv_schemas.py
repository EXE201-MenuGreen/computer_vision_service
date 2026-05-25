from pydantic import BaseModel, Field
from typing import List, Optional


# ── Detection ──────────────────────────────────────────────
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedFood(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    estimated_grams: float = Field(ge=0.0)


# ── Nutrition ──────────────────────────────────────────────
class MacroNutrients(BaseModel):
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: Optional[float] = None


class FoodNutrition(BaseModel):
    food_label: str
    estimated_grams: float
    macros: MacroNutrients
    usda_fdc_id: Optional[str] = None
    data_source: str = "unknown"
    # "verified" = admin-curated  | confidence 1.0
    # "pgvector"  = semantic DB hit | confidence 0.9
    # "usda"      = USDA API hit    | confidence 0.75
    # "fallback"  = hardcoded dict  | confidence 0.3
    # "unknown"   = reconstructed from history DB row
    confidence: float = 1.0


# ── CV Analysis Result ──────────────────────────────────────
class AnalysisResult(BaseModel):
    request_id: str
    detected_foods: List[DetectedFood]
    nutrition_breakdown: List[FoodNutrition]
    total_macros: MacroNutrients
    processing_time_ms: float


# ── Async job ──────────────────────────────────────────────
class JobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "Analysis queued. Poll /cv/jobs/{job_id} for result."


class JobStatusResponse(BaseModel):
    job_id: str
    status: str                   # queued | processing | done | failed
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None


# ── Health ──────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    version: str = "1.0.0"


# ── Meal History (Option 3) ─────────────────────────────────
class MealQueryRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)


class MealHistoryItem(BaseModel):
    id: str
    request_id: str
    analyzed_at: str
    foods: List[FoodNutrition]
    total_macros: MacroNutrients
    similarity: float


class MealQueryResponse(BaseModel):
    query: str
    results: List[MealHistoryItem]
    total_found: int
