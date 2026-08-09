# Task: Scan món ăn đã hoàn chỉnh

Quy ước trạng thái: `[ ]` chưa làm, `[/]` đang làm, `[x]` hoàn tất.

## Planner

- [x] Khảo sát router, worker, inference client, schemas, nutrition enrichment, cache và tests hiện tại.
- [x] Xác định API contract, ranh giới use case và tiêu chí tương thích ngược.
- [x] Lập `implementation_plan.md`, acceptance criteria, test plan và risk register.

## Developer

### Schema và shared utilities

- [x] Tạo `app/schemas/meal_scan_schemas.py` với raw/final ingredient, prepared-meal result và job status schemas.
- [x] Thêm validation confidence `[0,1]`, gram không âm, `analysis_type="prepared_meal"` và status literals.
- [x] Trích utility resize/JPEG encoding từ `cv_router.py` sang `image_validator.py`; giữ nguyên chất lượng/giới hạn ảnh hiện tại.
- [x] Bổ sung cache namespace với default tương thích cho luồng scan nguyên liệu.

### Provider và business logic

- [x] Refactor provider transport tối thiểu trong `inference_client.py`, không thay đổi hành vi/public entrypoint của luồng cũ.
- [x] Tạo `prepared_meal_service.py` và prompt riêng: nhận diện một món hoàn chỉnh, suy luận thành phần, gram và confidence; không gợi ý món khác.
- [x] Validate/normalize raw AI payload và tên nguyên liệu.
- [x] Gọi `NutritionService.lookup_batch` để dựng calories/macros từng nguyên liệu.
- [x] Tính `estimated_total_grams`, `total_macros`, source/confidence và estimation note.
- [x] Đảm bảo remote provider path prepared meal có capability rõ hoặc fail với lỗi cấu hình rõ ràng.

### Worker và API

- [x] Thêm task `cv.analyze_prepared_meal` và enqueue function riêng trong `worker.py`.
- [x] Áp dụng retry/circuit-breaker/logging conventions, kèm `analysis_type`; không dùng normalizer của response cũ.
- [x] Tạo `meal_scan_router.py` với `/cv/analyze-meal`, `/cv/analyze-meal-sync`, `/cv/meal-jobs/{job_id}`.
- [x] Dùng auth, validation ảnh, Celery job states và error handling hiện có.
- [x] Đăng ký router và cập nhật OpenAPI metadata trong `main.py`.
- [x] Không thay đổi request/response contract của ba endpoint scan nguyên liệu hiện tại.

### Documentation

- [x] Cập nhật `cv-service/README.md` với flow, curl và response mẫu.
- [x] Cập nhật `cv-service/docs/cv-service-guide.html` nếu đây vẫn là tài liệu phát hành chính.
- [x] Ghi rõ số liệu chỉ là ước tính, đặc biệt với dầu/sốt/đường và khẩu phần không có vật tham chiếu.

## Reviewer

- [x] Kiểm tra separation of concerns giữa ingredient scan và prepared-meal scan.
- [x] Kiểm tra API/schema backward compatibility của luồng hiện tại.
- [x] Kiểm tra calories cuối đến từ `NutritionService`, không tin calories do AI trả.
- [x] Kiểm tra cache namespace không collision/cross-contamination.
- [x] Kiểm tra auth, upload validation, log/error không lộ API key hoặc raw sensitive payload.
- [x] Kiểm tra duplicated async/sync/polling logic và refactor nếu cần mà không trộn response schemas.
- [x] Kiểm tra handling của hidden ingredients, confidence và estimation note.

## Tester

- [x] Viết/chạy schema boundary tests cho confidence, gram, required fields và literals.
- [x] Viết/chạy prompt/service tests cho prepared meal, normalization, USDA/fallback và aggregation.
- [x] Viết/chạy cache namespace isolation tests.
- [x] Viết/chạy worker task/retry/serialization tests.
- [x] Viết/chạy API tests: auth, invalid image, enqueue, poll mọi trạng thái, sync success/timeout, invalid AI payload.
- [x] Chạy regression tests cho `/cv/analyze`, `/cv/analyze-sync`, `/cv/jobs/{job_id}`.
- [x] Chạy toàn bộ `pytest` và ghi kết quả bàn giao.

## DevOps

- [ ] Build Docker image hiện tại sau thay đổi. **Blocked:** máy kiểm tra không có Docker CLI trong `PATH`; chưa thể chạy `docker compose config` hoặc build image thực tế.
- [ ] Khởi động Redis/API/worker và xác minh worker đăng ký `cv.analyze_image` cùng `cv.analyze_prepared_meal`. **Partial:** import registry đã xác minh cả hai task; startup container bị blocked do thiếu Docker CLI.
- [ ] Smoke test endpoint async + polling trong container. **Blocked:** thiếu Docker CLI; API/provider credentials local không được dùng để thay thế kiểm thử container.
- [x] Xác nhận không có env/secret mới bắt buộc ngoài cấu hình provider/nutrition hiện tại, hoặc document biến mới nếu phát sinh.

Kết quả kiểm chứng DevOps:

- `python -m compileall -q app`: PASS; `import app.main`: PASS.
- OpenAPI có đủ `POST /api/v1/cv/analyze-meal`, `POST /api/v1/cv/analyze-meal-sync`, `GET /api/v1/cv/meal-jobs/{job_id}`.
- Celery registry có `cv.analyze_image`, `cv.analyze_prepared_meal`, `cv.health_check`.
- Parse cấu trúc `docker-compose.yml` bằng PyYAML: PASS cho `cv-service`, `celery-worker`, `redis` và dependency health check; đây không thay thế `docker compose config`.
- `python -m pytest -q tests/test_meal_scan.py tests/test_prepared_meal_service.py tests/test_prepared_meal_worker.py tests/test_image_cache.py`: **52 passed**.
- `.env` local tồn tại và không được Git track; prepared-meal không thêm biến môi trường/secret mới. Credential readiness chỉ được kiểm tra dạng boolean, không ghi giá trị secret vào log.

## Reporter

- [x] Tạo `docs/report/walkthrough.md` mô tả file thay đổi, API examples, kết quả test/build và giới hạn độ chính xác.
- [x] Đối chiếu toàn bộ Acceptance Criteria trong `implementation_plan.md` trước khi báo hoàn tất.
