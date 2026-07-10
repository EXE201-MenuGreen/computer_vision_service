"""
Remote AI inference client.

Forwards images to Google Gemini or a remote API and returns structured JSON.
"""
from __future__ import annotations

import base64
import io
import json
import uuid
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cv_schemas import UserAnalysisContext
from app.services.image_cache import get_cached_result, hash_image, set_cached_result
from app.services.response_utils import ai_circuit_breaker

logger = get_logger(__name__)


class InferenceClientError(RuntimeError):
    """Error from inference client."""
    def __init__(self, *args, is_transient: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_transient = is_transient


def _build_personalization_block(context: Optional[UserAnalysisContext]) -> str:
    if context is None:
        return ""

    lines: list[str] = []
    if context.user_id:
        lines.append(f"- User ID: {context.user_id}")
    if context.dietary_preferences:
        lines.append(f"- Dietary preferences: {', '.join(context.dietary_preferences)}")
    if context.avoid_foods:
        lines.append(f"- Foods to avoid: {', '.join(context.avoid_foods)}")
    if context.recent_dishes:
        recent = ", ".join(context.recent_dishes[:10])
        lines.append(
            f"- Recent meals/ingredients this user already had (suggest DIFFERENT dishes): {recent}"
        )
    if context.allergies:
        lines.append(
            f"- Food allergies (NEVER use these allergens in any suggested dish): "
            f"{', '.join(context.allergies)}"
        )
    if context.health_conditions:
        lines.append(
            f"- Health conditions (avoid ingredients harmful for these): "
            f"{', '.join(context.health_conditions)}"
        )
    if context.dietary_goal:
        lines.append(f"- Dietary goal: {context.dietary_goal}")
    if context.avoid_ingredient_keys:
        lines.append(
            f"- Restricted ingredient keys: {', '.join(context.avoid_ingredient_keys)}"
        )
    if context.daily_calorie_limit:
        lines.append(
            f"- Daily calorie limit: {context.daily_calorie_limit} kcal — prefer lighter options when suitable"
        )

    if not lines:
        return ""

    safety_note = ""
    if context.allergies or context.health_conditions:
        safety_note = (
            "\n- CRITICAL: Every suggested dish must be safe for this user. "
            "Do not include allergic or health-restricted ingredients."
        )

    return (
        "\n\nPersonalization context (tailor suggestions accordingly):\n"
        + "\n".join(lines)
        + "\n- Prefer creative variety; do not repeat recent meals when alternatives exist."
        + safety_note
    )


def _build_gemini_prompt(context: Optional[UserAnalysisContext]) -> str:
    min_dishes = settings.gemini_min_dish_suggestions
    max_dishes = settings.gemini_max_dish_suggestions
    personalization = _build_personalization_block(context)

    return f"""
Phân tích hình ảnh món ăn/nguyên liệu này và trả về JSON theo cấu trúc sau.
Gợi ý từ {min_dishes} đến {max_dishes} món ăn KHÁC NHAU (phong cách, cách chế biến, hoặc mục tiêu dinh dưỡng khác nhau).
Mỗi món phải khả thi từ nguyên liệu nhìn thấy trong ảnh.
{personalization}
{{
  "api_version": "v1",
  "status": "done",
  "luong_tin_cay_chung": "độ tin cậy chung của phân tích (ví dụ: '92%')",
  "nguyen_lieu_tho_quet_duoc": [
    {{
      "id_nguyen_lieu": "mã tự sinh duy nhất, ví dụ: 'raw_01'",
      "ten_nguyen_lieu": "tên nguyên liệu thô bằng tiếng Việt",
      "ten_nguyen_lieu_ky_thuat": "mã snake_case tiếng Việt không dấu, ví dụ: 'uc_ga'",
      "khoi_luong_uoc_tinh_g": ước tính khối lượng gram (float),
      "do_chinh_xac_uoc_tinh": "độ chính xác ước tính (ví dụ: '95%')"
    }}
  ],
  "danh_sach_mon_an_goi_y": [
    {{
      "id_mon_an_goi_y": "mã tự sinh duy nhất, ví dụ: 'rec_01'",
      "ten_mon_an": "tên món ăn gợi ý bằng tiếng Việt",
      "ten_mon_an_ky_thuat": "mã snake_case tiếng Việt không dấu",
      "mo_ta_ngan": "mô tả ngắn về món ăn và cách làm sơ bộ",
      "do_kha_thi": "độ khả thi thực hiện (ví dụ: '95%')",
      "confidence": độ tin cậy float từ 0.0 đến 1.0,
      "nguyen_lieu_su_dung": [
        {{
          "ten": "tên nguyên liệu tiếng Việt",
          "ten_ky_thuat": "mã snake_case tiếng Việt không dấu",
          "khoi_luong_g": khối lượng gram (float)
        }}
      ],
      "thong_tin_dinh_duong_mon_an": {{
        "tong_calories": tổng calories kcal (float),
        "protein_g": protein gram (float),
        "carbs_g": carbs gram (float),
        "fat_g": fat gram (float),
        "fiber_g": fiber gram (float)
      }}
    }}
  ]
}}
Chỉ trả về duy nhất chuỗi JSON hợp lệ. Không dùng markdown (```json) hay văn bản giải thích.
"""


async def analyze_image_via_gemini(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    user_context: Optional[UserAnalysisContext] = None,
) -> dict[str, Any]:
    """Send an image to Gemini and return structured JSON."""
    if not settings.gemini_api_key:
        raise InferenceClientError("Gemini API key is not configured", is_transient=False)

    # Check circuit breaker before making API call
    if not ai_circuit_breaker.can_execute():
        logger.warning("gemini_circuit_breaker_open")
        raise InferenceClientError(
            "AI service temporarily unavailable (circuit breaker open)",
            is_transient=True,
        )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = content_type or "image/jpeg"
    prompt = _build_gemini_prompt(user_context)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": settings.gemini_temperature,
            "topP": settings.gemini_top_p,
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )
    timeout = httpx.Timeout(settings.ai_api_timeout_seconds)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # HTTP 5xx are transient, 4xx are not
                is_transient = 500 <= resp.status_code < 600
                ai_circuit_breaker.record_failure()
                logger.warning(
                    "gemini_api_http_error",
                    status_code=resp.status_code,
                    is_transient=is_transient,
                    body=resp.text[:500],
                )
                raise InferenceClientError(
                    f"Gemini API request failed: {exc}",
                    is_transient=is_transient,
                ) from exc

            data = resp.json()
            try:
                text_response = data["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text_response)
                if "job_id" not in result or not result["job_id"]:
                    result["job_id"] = str(uuid.uuid4())
                if "request_id" not in result or not result["request_id"]:
                    result["request_id"] = str(uuid.uuid4())

                # Record success for circuit breaker
                ai_circuit_breaker.record_success()
                return result
            except (KeyError, IndexError, json.JSONDecodeError) as err:
                # Malformed response is not transient - AI returned bad JSON
                logger.error(
                    "gemini_response_parse_failed",
                    error=str(err),
                    raw_response=resp.text[:1000],
                )
                raise InferenceClientError(
                    f"Failed to parse Gemini response: {err}",
                    is_transient=False,
                ) from err
    except httpx.TimeoutException:
        ai_circuit_breaker.record_failure()
        raise InferenceClientError("Gemini API timeout", is_transient=True)
    except httpx.ConnectError:
        ai_circuit_breaker.record_failure()
        raise InferenceClientError("Gemini API connection failed", is_transient=True)


async def analyze_image(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    user_context: Optional[UserAnalysisContext] = None,
) -> dict[str, Any]:
    """Main entrypoint: analyzes the image via Gemini or remote API.

    Image-hash cache: identical bytes return the cached response without
    contacting the AI provider. Disabled when image_cache_ttl_seconds=0.
    """
    if settings.image_cache_ttl_seconds > 0:
        image_hash = hash_image(image_bytes)
        cached = await get_cached_result(image_hash)
        if cached is not None:
            return cached

    if settings.ai_provider == "gemini":
        result = await analyze_image_via_gemini(
            image_bytes, filename, content_type, user_context
        )
    elif settings.ai_provider == "remote_api":
        if not settings.ai_api_base_url or not settings.ai_api_key:
            raise InferenceClientError("AI inference API is not configured", is_transient=False)

        # Check circuit breaker before making API call
        if not ai_circuit_breaker.can_execute():
            logger.warning("remote_api_circuit_breaker_open")
            raise InferenceClientError(
                "AI service temporarily unavailable (circuit breaker open)",
                is_transient=True,
            )

        timeout = httpx.Timeout(settings.ai_api_timeout_seconds)
        headers = {"Authorization": f"Bearer {settings.ai_api_key}"}
        files = {"image": (filename or "image.jpg", io.BytesIO(image_bytes), content_type or "image/jpeg")}
        data: dict[str, str] = {}
        if user_context is not None:
            data["user_context"] = user_context.model_dump_json()

        try:
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                resp = await client.post(
                    f"{settings.ai_api_base_url.rstrip('/')}/analyze",
                    files=files,
                    data=data,
                )
                try:
                    resp.raise_for_status()
                    ai_circuit_breaker.record_success()
                except httpx.HTTPStatusError as exc:
                    is_transient = 500 <= resp.status_code < 600
                    ai_circuit_breaker.record_failure()
                    logger.warning(
                        "ai_inference_http_error",
                        status_code=resp.status_code,
                        is_transient=is_transient,
                        body=resp.text[:500],
                    )
                    raise InferenceClientError(str(exc), is_transient=is_transient) from exc
                result = resp.json()
        except httpx.TimeoutException:
            ai_circuit_breaker.record_failure()
            raise InferenceClientError("Remote API timeout", is_transient=True)
        except httpx.ConnectError:
            ai_circuit_breaker.record_failure()
            raise InferenceClientError("Remote API connection failed", is_transient=True)
    else:
        raise InferenceClientError(
            f"Unsupported AI provider '{settings.ai_provider}'. Use 'gemini' or 'remote_api'.",
            is_transient=False,
        )

    if settings.image_cache_ttl_seconds > 0:
        await set_cached_result(image_hash, result)
    return result
