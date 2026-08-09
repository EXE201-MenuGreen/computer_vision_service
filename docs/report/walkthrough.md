# Walkthrough: Scan món ăn đã hoàn chỉnh

## 1. Kết quả bàn giao

`cv-service` đã có thêm một use case độc lập để phân tích ảnh **món ăn đã hoàn chỉnh**. Luồng mới nhận diện tên món, suy luận tối đa 20 nguyên liệu, ước lượng gram và độ tin cậy quan sát, sau đó tính calories/macros của từng nguyên liệu qua `NutritionService` và cộng thành tổng khẩu phần.

Ba endpoint scan nguyên liệu hiện hữu vẫn giữ nguyên contract. Luồng prepared-meal dùng prompt, schema, cache namespace, Celery task và endpoint riêng để không trộn payload hoặc business logic giữa hai use case.

## 2. Kiến trúc và luồng xử lý

```text
POST /api/v1/cv/analyze-meal
  -> Bearer auth
  -> validate MIME / dung lượng / decode ảnh
  -> resize và encode JPEG
  -> enqueue cv.analyze_prepared_meal
  -> Gemini nhận diện món + nguyên liệu + gram + detection confidence
  -> validate RawPreparedMealAnalysis
  -> normalize tên nguyên liệu
  -> NutritionService.lookup_batch (Redis -> USDA -> fallback)
  -> NutritionService.sum_macros
  -> validate PreparedMealAnalysisResponse
  -> GET /api/v1/cv/meal-jobs/{job_id}
```

Endpoint sync `/api/v1/cv/analyze-meal-sync` dùng cùng task và contract kết quả, chờ đến khi hoàn tất/thất bại hoặc trả `processing` khi hết thời gian chờ. Client production nên ưu tiên async và polling có backoff.

`remote_api` **chưa hỗ trợ prepared-meal**. Khi `AI_PROVIDER=remote_api`, service fail rõ bằng lỗi cấu hình không transient; hệ thống không âm thầm gửi prompt mới vào contract `/analyze` cũ. Gemini là provider được hỗ trợ cho luồng này ở phiên bản hiện tại.

## 3. Các file thay đổi

### File mới

| File | Vai trò |
|---|---|
| `cv-service/app/api/meal_scan_router.py` | Endpoint async, sync, polling và safe response mapping. |
| `cv-service/app/schemas/meal_scan_schemas.py` | Raw/final/job schemas; ràng buộc gram, confidence, literals và fan-out. |
| `cv-service/app/services/prepared_meal_service.py` | Prompt và orchestration riêng; normalize, nutrition enrichment, aggregate. |
| `cv-service/tests/test_meal_scan.py` | Auth, ảnh lỗi/MIME, enqueue, polling, sync, timeout và safe errors. |
| `cv-service/tests/test_prepared_meal_service.py` | Schema boundaries, prompt, cache, provider capability, USDA/fallback và aggregation. |
| `cv-service/tests/test_prepared_meal_worker.py` | Task registry, serialization, queue job id, retry và malformed payload. |
| `docs/task/implementation_plan.md` | Phạm vi, kiến trúc, acceptance criteria, test plan và risk register. |
| `docs/task/task.md` | Checklist phối hợp các vai trò trong `.agent`. |
| `docs/report/walkthrough.md` | Báo cáo nghiệm thu này. |

### File chỉnh sửa

| File | Thay đổi chính |
|---|---|
| `cv-service/app/main.py` | Đăng ký router và OpenAPI tag/description mới. |
| `cv-service/app/api/cv_router.py` | Dùng utility chuẩn bị ảnh chung; không đổi endpoint/contract cũ. |
| `cv-service/app/services/image_validator.py` | Tách resize/JPEG encoding thành utility dùng chung. |
| `cv-service/app/services/inference_client.py` | Tách Gemini transport theo prompt, thêm prepared-meal entrypoint/cache và capability check. |
| `cv-service/app/services/image_cache.py` | Thêm namespace tùy chọn, giữ key mặc định của ingredient scan. |
| `cv-service/app/services/worker.py` | Thêm `cv.analyze_prepared_meal`, enqueue, retry/circuit-breaker/log conventions. |
| `cv-service/tests/test_image_cache.py` | Kiểm tra namespace isolation và backward compatibility. |
| `cv-service/README.md` | Mô tả hai use case, endpoint, curl, response và accuracy caveats. |
| `cv-service/docs/cv-service-guide.html` | Bổ sung hướng dẫn prepared-meal vào tài liệu phát hành. |

Không có migration database, thay đổi Docker topology hoặc secret/env bắt buộc mới.

## 4. API examples

### Queue một job

```bash
curl -X POST "http://localhost:8000/api/v1/cv/analyze-meal" \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -F "image=@prepared-meal.jpg"
```

```json
{
  "job_id": "e54ad63b-9adb-4ef8-8239-b034e5b65892",
  "status": "queued",
  "message": "Prepared-meal analysis queued. Poll /cv/meal-jobs/{job_id} for result."
}
```

### Poll kết quả

```bash
curl "http://localhost:8000/api/v1/cv/meal-jobs/e54ad63b-9adb-4ef8-8239-b034e5b65892" \
  -H "Authorization: Bearer $API_SECRET_KEY"
```

Ví dụ response hoàn tất (rút gọn):

```json
{
  "job_id": "e54ad63b-9adb-4ef8-8239-b034e5b65892",
  "status": "done",
  "celery_state": "SUCCESS",
  "worker_active": true,
  "message": "Analysis completed successfully.",
  "steps": [],
  "result": {
    "job_id": "e54ad63b-9adb-4ef8-8239-b034e5b65892",
    "request_id": "67c50968-61e1-42b0-8bb1-493aaf4b9a88",
    "api_version": "v1",
    "status": "done",
    "analysis_type": "prepared_meal",
    "dish_name": "Cơm gà",
    "dish_name_key": "com_ga",
    "dish_confidence": 0.88,
    "estimated_total_grams": 420.0,
    "ingredients": [
      {
        "ingredient_id": "ingredient_01",
        "name": "Ức gà",
        "name_key": "uc_ga",
        "estimated_grams": 180.0,
        "detection_confidence": 0.82,
        "nutrition": {
          "macros": {
            "calories_kcal": 297.0,
            "protein_g": 55.8,
            "carbs_g": 0.0,
            "fat_g": 6.5,
            "fiber_g": 0.0
          },
          "data_source": "usda",
          "confidence": 0.75,
          "usda_fdc_id": "171077"
        }
      }
    ],
    "total_macros": {
      "calories_kcal": 609.0,
      "protein_g": 61.8,
      "carbs_g": 68.0,
      "fat_g": 8.0,
      "fiber_g": 1.2
    },
    "estimation_note": "Giá trị được ước tính từ hình ảnh và có thể thay đổi theo công thức, dầu, sốt, đường và khẩu phần thực tế."
  },
  "error": null
}
```

### Endpoint sync

```bash
curl -X POST "http://localhost:8000/api/v1/cv/analyze-meal-sync" \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -F "image=@prepared-meal.jpg" \
  -F "timeout_seconds=90"
```

Sync trả cùng `PreparedMealJobStatusResponse`; khi timeout, `status` là `processing` và client tiếp tục poll endpoint job.

## 5. Security và resilience

- Tất cả endpoint mới dùng Bearer auth hiện hữu và validation chung cho MIME, dung lượng, decode ảnh.
- Raw AI payload được validate trước khi fan-out sang nutrition lookup; NaN/Infinity, gram âm và confidence ngoài `[0,1]` bị từ chối.
- Danh sách AI sinh bị giới hạn tối đa 20 nguyên liệu để chặn provider-controlled USDA fan-out quá mức.
- Calories/macros do AI tự khai báo bị bỏ qua; số liệu cuối luôn được dựng lại từ `NutritionService`.
- HTTP body/raw response của Gemini không còn được ghi log trong các nhánh lỗi; client chỉ nhận thông báo lỗi prepared-meal an toàn, không nhận URL, API key hay provider payload.
- Payload hoàn tất được validate lại trước khi trả; payload sai schema chuyển thành trạng thái `failed` với lỗi chung.
- `job_id` trong nested result luôn được ghi đè bằng queue job id công khai, tránh metadata provider/cache làm lệch identity.
- Transient provider errors và circuit-breaker-open được retry; malformed AI JSON không retry.
- Cache `prepared_meal` tách khỏi cache ingredient scan; default key cũ được giữ để tương thích ngược.

## 6. Kết quả kiểm thử và kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Toàn bộ `pytest` | **PASS — 101 passed** |
| Prepared-meal/cache focused suite | **PASS — 52 passed** |
| `python -m compileall -q app` | **PASS** |
| `import app.main` | **PASS** |
| OpenAPI | **PASS** — có đủ 3 endpoint prepared-meal |
| Celery registry | **PASS** — có `cv.analyze_image`, `cv.analyze_prepared_meal`, `cv.health_check` |
| Parse `docker-compose.yml` bằng PyYAML | **PASS** — nhận diện `cv-service`, `celery-worker`, `redis` và dependency health check |
| Docker build/startup/container smoke test | **BLOCKED** — môi trường kiểm tra không có Docker CLI trong `PATH` |

Việc parse YAML **không thay thế** `docker compose config`, build image hay khởi động container. Vì vậy báo cáo này không tuyên bố Docker build pass.

## 7. Đối chiếu Acceptance Criteria

| # | Trạng thái | Đối chiếu |
|---:|---|---|
| 1 | **Pass** | Async endpoint dùng Bearer auth, validate ảnh, enqueue task riêng và trả `JobResponse` queued/job id; có API tests. |
| 2 | **Pass** | Polling map queued/processing/done/failed; kết quả done được validate bằng prepared-meal schema; có tests mọi trạng thái và invalid payload. |
| 3 | **Pass** | Sync trả final result hoặc `processing` với safe polling message khi timeout; có success/timeout tests. |
| 4 | **Pass** | Schema bắt buộc tên món, confidence, ít nhất một ingredient, gram không âm, nutrition breakdown và totals. |
| 5 | **Pass** | Service dùng `NutritionService.sum_macros`; total grams là tổng ingredient grams, có aggregation tests gồm USDA + fallback. |
| 6 | **Pass** | AI prompt cấm calories/macros; service chỉ dùng AI observation fields và dựng nutrition qua `lookup_batch`; test chứng minh AI calories giả bị bỏ qua. |
| 7 | **Pass** | Mỗi ingredient có `data_source`, nutrition confidence; final schema bắt buộc `estimation_note`. |
| 8 | **Pass** | Prepared-meal prompt yêu cầu một món hiện hữu, cấm suggestion; ingredient-scan entrypoint/prompt vẫn giữ mục đích cũ. |
| 9 | **Pass** | Cache mới dùng namespace `prepared_meal`; default ingredient key giữ nguyên; isolation/backward-compat tests pass. |
| 10 | **Pass** | Full suite 101 tests pass, bao gồm regression của các endpoint ingredient scan hiện hữu; refactor router cũ chỉ chuyển helper ảnh dùng chung. |
| 11 | **Pass** | Tests bao phủ auth, MIME/ảnh hỏng, invalid AI schema, timeout/retry; raw provider body và upstream error không bị trả cho client/log. |
| 12 | **Partial** | Toàn bộ pytest, compile/import, OpenAPI và Celery task registry đều pass. Docker image/worker startup và container smoke test bị **Blocked** do không có Docker CLI. |

Tổng hợp: **11 Pass, 1 Partial, 0 Blocked ở cấp toàn tiêu chí**. Phần chưa nghiệm thu của tiêu chí 12 là Docker runtime; từng bước Docker cụ thể vẫn mang trạng thái **Blocked**.

## 8. Giới hạn độ chính xác

- Kết quả là ước tính từ một ảnh, không phải phép đo calories hay khối lượng.
- Dầu, bơ, đường, nước sốt, gia vị và thành phần bị che có thể không nhìn thấy; prompt yêu cầu suy luận thận trọng và confidence thấp nhưng vẫn có sai số đáng kể.
- Khẩu phần khó ước lượng nếu ảnh không có vật tham chiếu kích thước, góc chụp bị nghiêng, món bị che hoặc nhiều món nằm chung ảnh.
- USDA matching cho nguyên liệu Việt Nam/món chế biến có thể rơi về fallback; UI/client nên hiển thị `data_source`, `detection_confidence`, nutrition `confidence` và `estimation_note`.
- V1 tối ưu cho một khẩu phần/một món chính, chưa tách riêng nhiều đĩa và chưa cho người dùng hiệu chỉnh gram/nguyên liệu sau scan.

## 9. Phần nghiệm thu còn lại

Trên máy có Docker CLI, cần chạy `docker compose config`, build image, khởi động `redis`, `cv-service`, `celery-worker`, xác minh worker đăng ký hai task phân tích và smoke test async + polling trong container. Cần cấu hình Gemini/API credentials hợp lệ để chạy E2E provider thật; không dùng `remote_api` cho prepared-meal cho đến khi upstream công bố contract/capability tương ứng.
