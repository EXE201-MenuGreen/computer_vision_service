# Food CV Microservice

Python FastAPI · YOLOv8 · EfficientNet · DepthAnything v2 · Docker

## Cấu trúc project

```
cv-service/
├── app/
│   ├── api/
│   │   └── cv_router.py        # FastAPI endpoints
│   ├── core/
│   │   ├── config.py           # Settings từ .env
│   │   └── logging.py          # Structured logging
│   ├── models/
│   │   └──   model_manager.py    # Singleton loader cho tất cả models
│   ├── schemas/
│   │   └── cv_schemas.py       # Pydantic request/response schemas
│   ├── services/
│   │   ├── cv_pipeline.py      # YOLO → classify → depth → grams
│   │   ├── image_validator.py  # MIME type, size, decode
│   │   ├── nutrition_service.py# USDA API lookup + fallback
│   │   └── worker.py           # Celery async job task
│   └── main.py                 # FastAPI app factory
├── tests/
│   └── test_cv_service.py
├── weights/                    # Đặt file .pt model tại đây
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Khởi động nhanh (Dành cho máy mới clone dự án)

Dự án này sử dụng công cụ [uv](https://github.com/astral-sh/uv) để quản lý môi trường ảo siêu tốc thay cho `pip` truyền thống.

**Bước 1: Clone dự án và tạo môi trường ảo**
```bash
git clone <your-repo-url>
cd cv-service

# Tạo virtual environment (.venv) siêu tốc bằng uv
uv venv

# Cài đặt tất cả thư viện từ requirements.txt
uv pip install -r requirements.txt
```

**Bước 2: Kích hoạt môi trường ảo**
- Trên Windows: `.venv\Scripts\activate`
- Trên macOS/Linux: `source .venv/bin/activate`

**Bước 3: Thiết lập biến môi trường**
Copy file mẫu và tùy chỉnh nếu cần:
```bash
cp .env.example .env
# Hoặc trên Windows PowerShell: copy .env.example .env
```

**Bước 4: Cài đặt Model Weights**
Tạo thư mục `weights/` và đặt file model `.pt` vào. (Nếu bỏ qua, hệ thống sẽ tự dùng model cơ bản làm fallback - xem bảng phía dưới).
```bash
mkdir weights
```

**Bước 5: Khởi chạy Server**
```bash
# Chạy FastAPI server ở chế độ dev (tự reload khi sửa code)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi chạy, server sẽ có mặt tại:
- API Docs: http://localhost:8000/docs
- Health check: http://localhost:8000/cv/health

## API Endpoints

| Method | Path | Mô tả |
|--------|------|--------|
| GET | `/cv/health` | Health check + model status |
| POST | `/cv/analyze` | Sync: upload → kết quả ngay (~2-5s) |
| POST | `/cv/analyze/async` | Async: trả job_id, dùng khi inference chậm |
| GET | `/cv/jobs/{job_id}` | Poll kết quả async job |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI (dev only) |

## Response mẫu

```json
{
  "request_id": "a1b2c3d4-...",
  "detected_foods": [
    {
      "label": "pho",
      "confidence": 0.923,
      "bbox": {"x1": 45, "y1": 80, "x2": 520, "y2": 430},
      "estimated_grams": 380.0
    }
  ],
  "nutrition_breakdown": [
    {
      "food_label": "pho",
      "estimated_grams": 380.0,
      "macros": {
        "calories_kcal": 817.0,
        "protein_g": 57.0,
        "carbs_g": 114.0,
        "fat_g": 15.2,
        "fiber_g": 2.1
      }
    }
  ],
  "total_macros": {
    "calories_kcal": 817.0,
    "protein_g": 57.0,
    "carbs_g": 114.0,
    "fat_g": 15.2,
    "fiber_g": 2.1
  },
  "processing_time_ms": 412.5
}
```

## Model Weights

Để chạy được chương trình, dự án cần các file model weights đặt trong thư mục `weights/`. Nếu chưa có, hãy tạo thư mục này:

```bash
mkdir -p weights
```

| File | Model | Tùy chọn tải (Fallback) |
|------|-------|-------------------------|
| `yolov8_food.pt` | YOLOv8 fine-tuned (Food-101) | Bạn có thể tự train. Nếu không có file này, chương trình sẽ tự động tải `yolov8n.pt` làm fallback. Bạn cũng có thể tải thủ công bằng lệnh: <br> `curl -L -o weights/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt` |
| `efficientnet_food.pt` | EfficientNet-B4 (Food-101) | Nếu không có, thư viện `timm` sẽ tự tải model pretrained cơ bản làm fallback. |

**Lưu ý:**
- Mô hình **Depth** (`DepthAnything-V2-Small`) sẽ được thư viện `transformers` tự động tải xuống từ HuggingFace trong lần đầu tiên bạn chạy service. Việc này có thể mất một vài phút tùy tốc độ mạng.

## GPU support

Sửa trong `.env`:
```
DEVICE=cuda
```

Bỏ comment phần `deploy.resources` trong `docker-compose.yml`.

## Chạy tests

```bash
pip install pytest pytest-asyncio
make test
```
