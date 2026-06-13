from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ── Detection ──────────────────────────────────────────────
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedFood(BaseModel):
    id_nguyen_lieu: str
    ten_nguyen_lieu_ky_thuat: str = Field(description="Canonical technical key, e.g. 'uc_ga'")
    ten_nguyen_lieu: str = Field(description="Vietnamese display label, e.g. 'Ức gà tươi sống'")
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
    food_label_key: str
    food_label_vi: str
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


# ── Analyze request context (personalization) ───────────────
class UserAnalysisContext(BaseModel):
    user_id: Optional[str] = None
    dietary_preferences: List[str] = Field(default_factory=list)
    avoid_foods: List[str] = Field(default_factory=list)
    recent_dishes: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list, description="Display names, e.g. Hải sản")
    allergy_keys: List[str] = Field(default_factory=list, description="allergen_key from backend DB")
    health_conditions: List[str] = Field(default_factory=list, description="Display names")
    health_condition_keys: List[str] = Field(default_factory=list)
    dietary_goal: Optional[str] = None
    avoid_ingredient_keys: List[str] = Field(default_factory=list)
    daily_calorie_limit: Optional[float] = None


# ── AI API response payload ────────────────────────────────
class IngredientItem(BaseModel):
    id_nguyen_lieu: str
    ten_nguyen_lieu: str
    ten_nguyen_lieu_ky_thuat: str
    khoi_luong_uoc_tinh_g: float
    do_chinh_xac_uoc_tinh: str


class RecipeIngredient(BaseModel):
    ten: str
    ten_ky_thuat: str
    khoi_luong_g: float


class NutritionInfo(BaseModel):
    tong_calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


class SuggestedDish(BaseModel):
    id_mon_an_goi_y: str
    ten_mon_an: str
    ten_mon_an_ky_thuat: Optional[str] = None
    mo_ta_ngan: str
    do_kha_thi: str
    confidence: float = Field(ge=0.0, le=1.0)
    nguyen_lieu_su_dung: List[RecipeIngredient]
    thong_tin_dinh_duong_mon_an: NutritionInfo
    an_toan_cho_user: Optional[bool] = None
    dich_ung_trung: Optional[List[str]] = Field(default_factory=list)
    canh_bao_suc_khoe: Optional[List[str]] = Field(default_factory=list)


class AIInferenceResponse(BaseModel):
    job_id: str
    request_id: str
    api_version: str = "v1"
    status: Literal["queued", "processing", "done", "failed"]
    processing_time_ms: Optional[float] = None
    luong_tin_cay_chung: Optional[str] = None
    nguyen_lieu_tho_quet_duoc: Optional[List[IngredientItem]] = None
    danh_sach_mon_an_goi_y: Optional[List[SuggestedDish]] = None
    mon_an_goi_y_chon: Optional[SuggestedDish] = None
    nutrition_breakdown: Optional[List[FoodNutrition]] = None
    total_macros: Optional[MacroNutrients] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


# ── Async job ──────────────────────────────────────────────
class JobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "Analysis queued. Poll /cv/jobs/{job_id} for result."


class JobStatusResponse(BaseModel):
    job_id: str
    status: str                   # queued | processing | done | failed
    result: Optional[AIInferenceResponse] = None
    error: Optional[str] = None


# ── Health ──────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    redis: bool
    worker: bool
    gemini_configured: bool
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
