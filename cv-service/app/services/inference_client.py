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

logger = get_logger(__name__)


class InferenceClientError(RuntimeError):
    pass


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

    if not lines:
        return ""

    return (
        "\n\nPersonalization context (tailor suggestions accordingly):\n"
        + "\n".join(lines)
        + "\n- Prefer creative variety; do not repeat recent meals when alternatives exist."
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
        raise InferenceClientError("Gemini API key is not configured")

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

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("gemini_api_http_error", status_code=resp.status_code, body=resp.text[:500])
            raise InferenceClientError(f"Gemini API request failed: {exc}") from exc

        data = resp.json()
        try:
            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            result = json.loads(text_response)
            if "job_id" not in result or not result["job_id"]:
                result["job_id"] = str(uuid.uuid4())
            if "request_id" not in result or not result["request_id"]:
                result["request_id"] = str(uuid.uuid4())
            return result
        except (KeyError, IndexError, json.JSONDecodeError) as err:
            logger.error("gemini_response_parse_failed", error=str(err), raw_response=resp.text[:1000])
            raise InferenceClientError(f"Failed to parse Gemini response: {err}") from err


async def analyze_image(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    user_context: Optional[UserAnalysisContext] = None,
) -> dict[str, Any]:
    """Main entrypoint: analyzes the image via Gemini or remote API."""
    if settings.ai_provider == "gemini":
        return await analyze_image_via_gemini(image_bytes, filename, content_type, user_context)

    if settings.ai_provider == "remote_api":
        if not settings.ai_api_base_url or not settings.ai_api_key:
            raise InferenceClientError("AI inference API is not configured")

        timeout = httpx.Timeout(settings.ai_api_timeout_seconds)
        headers = {"Authorization": f"Bearer {settings.ai_api_key}"}
        files = {"image": (filename or "image.jpg", io.BytesIO(image_bytes), content_type or "image/jpeg")}
        data: dict[str, str] = {}
        if user_context is not None:
            data["user_context"] = user_context.model_dump_json()

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            resp = await client.post(
                f"{settings.ai_api_base_url.rstrip('/')}/analyze",
                files=files,
                data=data,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning("ai_inference_http_error", status_code=resp.status_code, body=resp.text[:500])
                raise InferenceClientError(str(exc)) from exc
            return resp.json()

    raise InferenceClientError(
        f"Unsupported AI provider '{settings.ai_provider}'. Use 'gemini' or 'remote_api'."
    )
