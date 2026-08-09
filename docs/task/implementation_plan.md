# Kế hoạch triển khai: Scan món ăn đã hoàn chỉnh

## 1. Mục tiêu

Bổ sung một use case độc lập trong `cv-service` để người dùng tải ảnh **món ăn đã hoàn chỉnh**, nhận về:

- tên món ăn và độ tin cậy nhận diện;
- danh sách nguyên liệu thành phần được suy luận từ ảnh;
- khối lượng ước tính và độ tin cậy quan sát của từng nguyên liệu;
- calories, protein, carbs, fat và fiber của từng nguyên liệu;
- tổng khối lượng và tổng calories/macros của khẩu phần;
- nguồn dữ liệu và độ tin cậy dinh dưỡng, cùng cảnh báo đây là số liệu ước tính.

Luồng hiện tại `/api/v1/cv/analyze`, `/api/v1/cv/analyze-sync` và `/api/v1/cv/jobs/{job_id}` tiếp tục phục vụ việc scan nguyên liệu rồi gợi ý món, không thay đổi request/response contract.

## 2. Phạm vi và quyết định kiến trúc

### 2.1. Tách use case, dùng chung hạ tầng

Luồng mới được tách theo chiều dọc:

```text
POST /api/v1/cv/analyze-meal
        |
        v
meal_scan_router -> cv.analyze_prepared_meal -> prepared_meal_service
        |                                      |
        |                                      +-> Gemini/remote AI: nhận diện món + suy luận thành phần
        |                                      +-> NutritionService: tính macros theo gram
        v
GET /api/v1/cv/meal-jobs/{job_id} -> PreparedMealJobStatusResponse
```

Các thành phần dùng chung: xác thực API key, validation/resize ảnh, Celery/Redis, circuit breaker, provider configuration, `NutritionService`, food-label normalization và job progress conventions.

### 2.2. API contract dự kiến

Endpoint bất đồng bộ chính:

```http
POST /api/v1/cv/analyze-meal
Content-Type: multipart/form-data
Authorization: Bearer <token>

image=<file>
```

Response ban đầu dùng `JobResponse`. Client poll endpoint riêng:

```http
GET /api/v1/cv/meal-jobs/{job_id}
```

Để đồng nhất khả năng hiện tại, bổ sung endpoint sync:

```http
POST /api/v1/cv/analyze-meal-sync
```

Kết quả hoàn tất dự kiến:

```json
{
  "job_id": "...",
  "request_id": "...",
  "api_version": "v1",
  "status": "done",
  "analysis_type": "prepared_meal",
  "dish_name": "Cơm gà xối mỡ",
  "dish_name_key": "com_ga_xoi_mo",
  "dish_confidence": 0.81,
  "estimated_total_grams": 415.0,
  "ingredients": [
    {
      "ingredient_id": "ingredient_01",
      "name": "Cơm trắng",
      "name_key": "com_trang",
      "estimated_grams": 220.0,
      "detection_confidence": 0.86,
      "nutrition": {
        "macros": {
          "calories_kcal": 286.0,
          "protein_g": 5.9,
          "carbs_g": 62.0,
          "fat_g": 0.7,
          "fiber_g": 0.9
        },
        "data_source": "usda",
        "confidence": 0.75,
        "usda_fdc_id": "..."
      }
    }
  ],
  "total_macros": {
    "calories_kcal": 815.0,
    "protein_g": 42.0,
    "carbs_g": 70.0,
    "fat_g": 39.0,
    "fiber_g": 3.0
  },
  "estimation_note": "Giá trị được ước tính từ hình ảnh và có thể thay đổi theo công thức, dầu, sốt và khẩu phần thực tế."
}
```

`detection_confidence` phản ánh khả năng suy luận từ ảnh; `nutrition.confidence` phản ánh chất lượng nguồn dữ liệu dinh dưỡng. Hai trường không được gộp vì có ý nghĩa khác nhau.

### 2.3. Quy tắc tính dinh dưỡng

1. AI chỉ chịu trách nhiệm nhận diện tên món, suy luận nguyên liệu, ước lượng gram và confidence.
2. Tên nguyên liệu được chuẩn hóa bằng `food_labels.normalize_ingredient`.
3. `NutritionService.lookup_batch` lấy macros/100 g theo Redis -> USDA -> fallback và scale theo gram.
4. Calories/macros trả cho từng nguyên liệu lấy từ `NutritionService`, không tin trực tiếp giá trị calories do AI sinh.
5. `NutritionService.sum_macros` tạo tổng; `estimated_total_grams` là tổng gram của các nguyên liệu đã chuẩn hóa.
6. Dầu, đường, bơ và nước sốt không nhìn thấy rõ phải được prompt yêu cầu suy luận thận trọng, confidence thấp và không được trình bày như phép đo chính xác.

### 2.4. Cache và tính tương thích

Cache ảnh hiện chỉ dùng SHA-256 của bytes. Cùng một ảnh chạy hai use case có thể trả nhầm kiểu payload nếu dùng chung key. Cache phải hỗ trợ namespace, ví dụ:

```text
image_result:ingredient_scan:<sha256>
image_result:prepared_meal:<sha256>
```

Namespace mặc định của API cache hiện tại phải giữ hành vi tương thích cho luồng cũ. Test cần chứng minh cùng bytes nhưng khác namespace không va chạm.

## 3. Danh sách file bị tác động

### API và schema

- `[NEW] cv-service/app/api/meal_scan_router.py`
  - Khai báo endpoint async, sync và polling riêng.
  - Dùng `require_api_key`, image validation và chuẩn tiến độ job hiện có.
  - Validate payload hoàn tất bằng schema prepared-meal; lỗi schema trả trạng thái `failed` có thông báo an toàn.
- `[NEW] cv-service/app/schemas/meal_scan_schemas.py`
  - Khai báo raw AI result, ingredient nutrition result, prepared-meal result và job status response.
  - Áp ràng buộc gram không âm, confidence trong `[0, 1]`, status/analysis type bằng `Literal`.
- `[MODIFY] cv-service/app/main.py`
  - Include router mới và cập nhật mô tả OpenAPI/tag để thể hiện hai use case.
- `[MODIFY] cv-service/app/api/cv_router.py`
  - Chỉ refactor phần helper dùng chung nếu cần; không đổi endpoint hay response schema hiện tại.
- `[MODIFY] cv-service/app/services/image_validator.py`
  - Chuyển/expose bước resize và JPEG encoding thành utility dùng chung để hai router không duplicate.

### Business logic và worker

- `[NEW] cv-service/app/services/prepared_meal_service.py`
  - Sở hữu prompt và orchestration riêng cho món hoàn chỉnh.
  - Gọi provider transport, normalize nguyên liệu, tra dinh dưỡng, cộng macros và dựng response cuối.
  - Không sinh danh sách món gợi ý và không gọi logic chọn món ngẫu nhiên của use case cũ.
- `[MODIFY] cv-service/app/services/inference_client.py`
  - Tách phần gửi request provider dùng chung khỏi prompt use case hiện tại hoặc cung cấp transport nhận prompt/schema mục tiêu.
  - Giữ `analyze_image(...)` và prompt scan nguyên liệu tương thích ngược.
  - Remote provider phải nhận được loại phân tích/route tương ứng; nếu upstream chưa hỗ trợ prepared meal thì trả lỗi cấu hình rõ ràng, không âm thầm dùng contract cũ.
- `[MODIFY] cv-service/app/services/image_cache.py`
  - Thêm cache namespace có default tương thích; log kèm namespace.
- `[MODIFY] cv-service/app/services/worker.py`
  - Thêm task `cv.analyze_prepared_meal` và hàm enqueue riêng.
  - Task gọi `prepared_meal_service`, giữ retry/circuit-breaker semantics và không đi qua `normalize_ai_response` dành cho schema cũ.
  - Tái sử dụng job-result reader nếu không cần thay đổi trạng thái Celery.

### Tests và tài liệu

- `[NEW] cv-service/tests/test_meal_scan.py`
  - Test auth, validation ảnh, enqueue, async polling, sync timeout/success, schema lỗi và giữ đúng response contract mới.
- `[NEW] cv-service/tests/test_prepared_meal_service.py`
  - Test prompt không yêu cầu gợi ý món; normalize tên; calories/macros từng nguyên liệu; tổng; nguồn fallback; nguyên liệu ẩn và confidence.
- `[MODIFY] cv-service/tests/test_image_cache.py`
  - Test namespace tách cache giữa hai use case và backward compatibility.
- `[MODIFY] cv-service/tests/test_cv_service.py`
  - Regression test xác nhận endpoint scan nguyên liệu hiện tại vẫn giữ nguyên hành vi.
- `[MODIFY] cv-service/README.md`
  - Document hai use case, endpoint, ví dụ request/response và cảnh báo độ chính xác.
- `[MODIFY] cv-service/docs/cv-service-guide.html`
  - Cập nhật hướng dẫn API/flow nếu tài liệu này vẫn là tài liệu phát hành chính.

Không cần migration database hoặc thay đổi Docker topology. Celery worker hiện load module `app.services.worker`, nên task mới được đăng ký cùng worker hiện tại.

## 4. Trình tự triển khai

1. Định nghĩa schema raw/final/job cho prepared meal và fixture payload chuẩn.
2. Tách utility xử lý ảnh dùng chung, bảo toàn tests của router cũ.
3. Namespace image cache và bổ sung regression tests.
4. Tách provider transport dùng chung; giữ nguyên public function của luồng cũ.
5. Cài đặt `prepared_meal_service`: prompt -> validate raw result -> normalize -> nutrition lookup -> aggregate.
6. Thêm Celery task/enqueue riêng với retry và logging có `analysis_type`.
7. Thêm router async/sync/polling và đăng ký trong `main.py`.
8. Viết unit/API tests; chạy toàn bộ suite để phát hiện regression.
9. Cập nhật README và service guide.
10. Reviewer kiểm tra contract, security, cache isolation và duplicated logic; Tester/DevOps nghiệm thu test/build.

## 5. Acceptance Criteria

1. `POST /api/v1/cv/analyze-meal` nhận ảnh hợp lệ, yêu cầu Bearer token và trả `job_id` ở trạng thái `queued`.
2. `GET /api/v1/cv/meal-jobs/{job_id}` trả đúng các trạng thái queued/processing/done/failed và khi done validate bằng schema prepared-meal.
3. Endpoint sync mới trả kết quả cuối hoặc trạng thái processing khi timeout, cùng semantics với endpoint sync hiện tại.
4. Kết quả done có tên món, confidence, danh sách ít nhất một nguyên liệu khi AI nhận diện được, gram không âm, macros từng nguyên liệu và `total_macros`.
5. `total_macros` bằng tổng breakdown theo quy tắc làm tròn hiện tại của `NutritionService`; `estimated_total_grams` bằng tổng gram nguyên liệu.
6. Calories/macros cuối được tính qua `NutritionService` (USDA/fallback), không lấy thẳng số calories AI tự khai báo.
7. Mỗi nguyên liệu nêu rõ `data_source` và nutrition confidence; kết quả luôn có `estimation_note`.
8. Prompt prepared-meal không yêu cầu gợi ý 3–5 món và prompt scan nguyên liệu hiện tại không bị thay đổi về mục đích.
9. Cùng một ảnh ở hai use case không dùng nhầm cache payload.
10. `/api/v1/cv/analyze`, `/api/v1/cv/analyze-sync`, `/api/v1/cv/jobs/{job_id}` vẫn pass regression tests và giữ response contract cũ.
11. Upload sai MIME/ảnh hỏng, thiếu/sai token, AI JSON lỗi và provider timeout được xử lý theo convention hiện tại; không lộ secret hoặc raw provider response cho client.
12. Toàn bộ `pytest` pass và Docker image/worker khởi động, task `cv.analyze_prepared_meal` được đăng ký.

## 6. Kế hoạch kiểm thử

### Unit tests

- Schema boundary: confidence -0.1/1.1, gram âm, thiếu tên món, status sai.
- Prompt: prepared meal yêu cầu một món hiện hữu và thành phần, không có danh sách suggestion.
- Enrichment: known label, English alias, unknown label fallback, zero gram, fiber `None`.
- Aggregation: tổng calories/macros và gram đúng sau rounding.
- Cache: hit/miss theo namespace, Redis error vẫn fail-open như hiện tại.
- Worker: task name, serialization, retry transient và không retry malformed AI JSON.

### API/integration tests có mock

- Auth 401/429, multipart validation, async enqueue.
- Poll queued/processing/done/failed.
- Sync done và timeout.
- AI payload hợp lệ/thiếu trường/sai kiểu.
- Nutrition lookup mock USDA/fallback và schema response cuối.

### Regression và build

- Chạy `pytest` toàn suite.
- Khởi động FastAPI và kiểm tra OpenAPI có cả hai nhóm endpoint.
- Build Docker image và kiểm tra worker đăng ký hai task phân tích.

## 7. Rủi ro và biện pháp giảm thiểu

| Rủi ro | Mức độ | Giảm thiểu |
|---|---:|---|
| Không thể xác định chính xác dầu, đường, sốt hoặc thành phần bị che | Cao | Prompt yêu cầu suy luận thận trọng; tách detection confidence; luôn trả estimation note; về sau cho client chỉnh gram/thành phần. |
| Ước lượng khẩu phần từ một ảnh sai | Cao | Không gọi là phép đo; trả gram/confidence; khuyến nghị ảnh rõ và có vật tham chiếu trong docs. |
| Cache cùng bytes trả nhầm response của use case kia | Cao | Namespace cache theo analysis type và test isolation. |
| USDA match sai nguyên liệu Việt Nam/món chế biến | Trung bình-cao | Chuẩn hóa label, giữ name-match threshold, source/confidence, fallback rõ ràng; không tra tên cả món thay cho từng thành phần. |
| Refactor provider làm regression luồng hiện tại | Trung bình | Giữ public API cũ, golden prompt/regression tests, thay đổi từng bước. |
| Remote AI provider chỉ hỗ trợ endpoint `/analyze` contract cũ | Trung bình | Quy định capability/route prepared meal rõ ràng; fail rõ nếu không hỗ trợ; Gemini là path chuẩn ban đầu. |
| Duplicate logic async/sync/polling giữa routers | Trung bình | Trích helper nội bộ dùng chung khi contract giống nhau, nhưng giữ schema validation theo use case. |
| Circuit breaker dùng chung khiến một luồng ảnh hưởng luồng kia | Thấp-trung bình | Log `analysis_type`; chấp nhận breaker dùng chung provider ở phiên bản đầu, theo dõi để tách nếu vận hành cần. |
| Cost/latency tăng do AI + nhiều USDA lookup tuần tự | Trung bình | Tận dụng cache; giới hạn số nguyên liệu hợp lý; cân nhắc parallel lookup có giới hạn ở bước tối ưu sau khi đo. |

## 8. Breaking changes và migration

- Không có breaking API change chủ đích đối với luồng hiện tại.
- Chỉ thêm endpoint, schema và Celery task.
- Thay đổi key cache phải giữ default key/khả năng đọc luồng cũ hoặc chấp nhận cache miss có kiểm soát; không yêu cầu data migration vì cache có TTL.
- Không thay đổi database schema.

## 9. Open Questions

Các giả định dưới đây đủ an toàn để triển khai phiên bản đầu, nhưng Product Owner có thể điều chỉnh sau:

1. Endpoint sync được thêm để đối xứng API hiện tại; client production nên ưu tiên async.
2. Prepared-meal v1 phân tích một khẩu phần/một món chính trong ảnh. Ảnh có nhiều món sẽ trả tên tổng quát và ingredients chung, chưa tách nhiều đĩa.
3. Không lưu lịch sử prepared meal trong phạm vi này; chỉ trả kết quả phân tích. Việc persist cần một task riêng vì schema lịch sử hiện dựa trên `FoodNutrition` của luồng cũ.
4. User personalization không cần để nhận diện món; v1 không dùng dietary preferences để sửa sự thật quan sát được. Allergy/health warning có thể bổ sung sau mà không chi phối danh sách nguyên liệu.

