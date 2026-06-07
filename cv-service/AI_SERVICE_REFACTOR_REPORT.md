# AI Service Refactor Report

## Mục tiêu đã thống nhất
- Backend là điểm vào duy nhất cho client.
- Backend truyền ảnh sang AI service.
- AI service xác thực bằng `Authorization: Bearer <key>`.
- AI service xử lý bất đồng bộ để tránh treo request.
- AI service trả JSON theo contract chuẩn hóa.
- Backend nhận JSON và xử lý business logic cuối cùng.

---

## Quy trình refactor đã thực hiện

### 1) Chuyển kiến trúc từ local weights sang API-based
Đã cập nhật cấu hình để chuẩn bị cho AI API remote:
- thêm `ai_api_base_url`
- thêm `ai_api_key`
- thêm `ai_api_timeout_seconds`
- thêm `ai_api_poll_interval_seconds`

Đồng thời tạo lớp client gọi AI API ngoài:
- `app/services/inference_client.py`

---

### 2) Chuẩn hóa cơ chế xác thực bằng API key
Đã thêm helper xác thực API key cho backend → AI service:
- `Authorization: Bearer <key>`
- file mới: `app/api/auth.py`

Router AI hiện đã gắn dependency xác thực API key.

---

### 3) Chuyển luồng xử lý sang bất đồng bộ
Đã đổi endpoint phân tích ảnh sang mô hình async job:
- `POST /api/v1/cv/analyze` trả `job_id`
- `GET /api/v1/cv/jobs/{job_id}` để poll trạng thái/kết quả

Worker hiện không còn phụ thuộc local weights/pipeline và chuyển sang gọi AI API ngoài.

---

### 4) Chuẩn hóa schema response
Đã mở rộng schema để phục vụ contract mới:
- nguyên liệu thô
- món ăn gợi ý
- thông tin dinh dưỡng
- job status
- metadata API

Các model chính hiện có:
- `AIInferenceResponse`
- `IngredientItem`
- `SuggestedDish`
- `RecipeIngredient`
- `NutritionInfo`

---

### 5) Chuyển AI service sang vai trò đúng
AI service hiện được thiết kế để:
- nhận ảnh từ backend
- gọi AI API ngoài bằng Bearer token
- trả JSON chuẩn hóa
- không còn đóng vai trò backend nghiệp vụ chính

---

## Contract response chuẩn hiện tại

### Response cấp cao
```json
{
  "job_id": "job_123",
  "request_id": "req_123",
  "api_version": "v1",
  "status": "done",
  "processing_time_ms": 1842,
  "luong_tin_cay_chung": "92%",
  "nguyen_lieu_tho_quet_duoc": [],
  "danh_sach_mon_an_goi_y": []
}
```

### Trạng thái job
- `queued`
- `processing`
- `done`
- `failed`

### Lỗi chuẩn hóa
- `error_code`
- `error_message`

---

## Quy tắc label đã thống nhất

### 1) Tách 2 lớp label
- `label_key` / `ten_nguyen_lieu_ky_thuat`: mã kỹ thuật ổn định
- `label_vi` / `ten_nguyen_lieu`: label tiếng Việt để hiển thị

### 2) Mỗi label kỹ thuật chỉ map ra một label tiếng Việt chuẩn
Ví dụ:
- `uc_ga` → `Ức gà tươi sống`
- `bong_cai_xanh` → `Bông cải xanh (Súp lơ)`

### 3) AI không tự do sinh label
AI/API nên trả label theo contract hoặc canonical key, sau đó service map sang label tiếng Việt.

---

## Tình trạng kỹ thuật hiện tại

### Đã làm xong
- tạo lớp gọi AI API ngoài
- tạo auth API key cho AI service
- chuyển endpoint phân tích ảnh sang job-based async flow
- chuẩn hóa schema response
- chuẩn bị nền để bỏ phụ thuộc model local

### Còn cần hoàn thiện nếu muốn production hóa hoàn toàn
- dọn sạch toàn bộ logic local pipeline còn sót
- thay job store in-memory bằng Redis/DB nếu cần production
- hoàn thiện luồng backend ↔ AI service ↔ AI API ngoài
- viết test mới theo contract API-based
- rà lại các router/utility còn sót logic cũ

---

## Ghi chú quan trọng
- AI service hiện đã chuyển đúng hướng kiến trúc, nhưng một số file cũ vẫn còn dấu vết logic local/model weights.
- Nếu mục tiêu là production hoàn chỉnh, cần thêm một vòng cleanup cuối cùng để loại bỏ hoàn toàn các nhánh local inference cũ.

---

## Kết luận
Kiến trúc mục tiêu hiện tại đã rõ:
- Backend gọi AI service
- AI service xác thực Bearer token
- AI service xử lý ảnh bất đồng bộ
- AI service trả JSON chuẩn hóa
- Backend xử lý logic cuối cùng

File này là bản ghi nhận tiến trình refactor và định hướng triển khai tiếp theo.
