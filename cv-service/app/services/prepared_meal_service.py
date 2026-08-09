from __future__ import annotations

import uuid
from typing import Any

from app.schemas.cv_schemas import BoundingBox, DetectedFood
from app.schemas.meal_scan_schemas import (
    PreparedMealAnalysisResponse,
    PreparedMealIngredient,
    PreparedMealNutrition,
    RawPreparedMealAnalysis,
)
from app.services.food_labels import normalize_ingredient
from app.services.inference_client import analyze_prepared_meal_image
from app.services.nutrition_service import nutrition_service

ESTIMATION_NOTE = (
    "Giá trị được ước tính từ hình ảnh và có thể thay đổi theo công thức, "
    "dầu, sốt, đường và khẩu phần thực tế."
)


def build_prepared_meal_prompt() -> str:
    return """
Phân tích MỘT MÓN ĂN ĐÃ HOÀN CHỈNH đang hiện hữu trong ảnh.
Nhận diện tên món và suy luận thận trọng các nguyên liệu cấu thành, kể cả dầu,
đường, bơ hoặc nước sốt có khả năng đã dùng nhưng không nhìn thấy rõ. Các thành
phần ẩn phải có detection_confidence thấp. Chỉ ước lượng gram và confidence;
chỉ trả tối đa 20 nguyên liệu quan trọng nhất;
KHÔNG trả calories/macros và KHÔNG gợi ý món ăn khác.
Trả duy nhất JSON hợp lệ theo cấu trúc:
{
  "dish_name": "tên món tiếng Việt",
  "dish_name_key": "snake_case_khong_dau",
  "dish_confidence": 0.0,
  "ingredients": [
    {
      "ingredient_id": "ingredient_01",
      "name": "tên nguyên liệu",
      "name_key": "canonical_or_snake_case_key",
      "estimated_grams": 0.0,
      "detection_confidence": 0.0
    }
  ]
}
""".strip()


async def analyze_prepared_meal(
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> dict[str, Any]:
    raw_payload = await analyze_prepared_meal_image(
        image_bytes, filename, content_type, build_prepared_meal_prompt()
    )
    # Ignore any nutrition fields emitted by the model by validating only the
    # observation contract and rebuilding all nutrition below.
    raw = RawPreparedMealAnalysis.model_validate(raw_payload)

    detected: list[DetectedFood] = []
    normalized_rows = []
    for item in raw.ingredients:
        label = normalize_ingredient(item.name_key, item.name)
        normalized_rows.append((item, label))
        detected.append(
            DetectedFood(
                id_nguyen_lieu=item.ingredient_id,
                ten_nguyen_lieu_ky_thuat=label.key,
                ten_nguyen_lieu=label.vi_display,
                confidence=item.detection_confidence,
                bbox=BoundingBox(x1=0, y1=0, x2=0, y2=0),
                estimated_grams=item.estimated_grams,
            )
        )

    breakdown = await nutrition_service.lookup_batch(detected)
    if len(breakdown) != len(normalized_rows):
        raise RuntimeError("Nutrition lookup returned an incomplete ingredient breakdown")
    ingredients = [
        PreparedMealIngredient(
            ingredient_id=item.ingredient_id,
            name=label.vi_display,
            name_key=label.key,
            estimated_grams=item.estimated_grams,
            detection_confidence=item.detection_confidence,
            nutrition=PreparedMealNutrition(
                macros=nutrition.macros,
                data_source=nutrition.data_source,
                confidence=nutrition.confidence,
                usda_fdc_id=nutrition.usda_fdc_id,
            ),
        )
        for (item, label), nutrition in zip(normalized_rows, breakdown)
    ]

    return PreparedMealAnalysisResponse(
        job_id=str(raw_payload.get("job_id") or uuid.uuid4()),
        request_id=str(raw_payload.get("request_id") or uuid.uuid4()),
        dish_name=raw.dish_name,
        dish_name_key=raw.dish_name_key,
        dish_confidence=raw.dish_confidence,
        estimated_total_grams=round(sum(i.estimated_grams for i in ingredients), 1),
        ingredients=ingredients,
        total_macros=nutrition_service.sum_macros(breakdown),
        estimation_note=ESTIMATION_NOTE,
    ).model_dump()
