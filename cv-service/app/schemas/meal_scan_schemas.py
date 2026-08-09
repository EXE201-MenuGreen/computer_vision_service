from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.cv_schemas import JobProgressStep, MacroNutrients


class RawPreparedMealIngredient(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    ingredient_id: str
    name: str = Field(min_length=1)
    name_key: str = Field(min_length=1)
    estimated_grams: float = Field(ge=0.0)
    detection_confidence: float = Field(ge=0.0, le=1.0)


class RawPreparedMealAnalysis(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    dish_name: str = Field(min_length=1)
    dish_name_key: str = Field(min_length=1)
    dish_confidence: float = Field(ge=0.0, le=1.0)
    # Bound provider-controlled fan-out: every row can trigger a USDA lookup.
    ingredients: List[RawPreparedMealIngredient] = Field(min_length=1, max_length=20)


class PreparedMealNutrition(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    macros: MacroNutrients
    data_source: str
    confidence: float = Field(ge=0.0, le=1.0)
    usda_fdc_id: Optional[str] = None


class PreparedMealIngredient(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    ingredient_id: str
    name: str
    name_key: str
    estimated_grams: float = Field(ge=0.0)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    nutrition: PreparedMealNutrition


class PreparedMealAnalysisResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    job_id: str
    request_id: str
    api_version: Literal["v1"] = "v1"
    status: Literal["done"] = "done"
    analysis_type: Literal["prepared_meal"] = "prepared_meal"
    dish_name: str = Field(min_length=1)
    dish_name_key: str = Field(min_length=1)
    dish_confidence: float = Field(ge=0.0, le=1.0)
    estimated_total_grams: float = Field(ge=0.0)
    ingredients: List[PreparedMealIngredient] = Field(min_length=1)
    total_macros: MacroNutrients
    estimation_note: str = Field(min_length=1)


class PreparedMealJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "failed"]
    celery_state: Optional[str] = None
    worker_active: bool = False
    message: str
    steps: List[JobProgressStep] = Field(default_factory=list)
    result: Optional[PreparedMealAnalysisResponse] = None
    error: Optional[str] = None
