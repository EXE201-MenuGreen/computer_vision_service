Dockerfile này của bạn đã được cải tiến rất nhiều so với bản đầu tiên! Việc thêm các biến môi trường (`PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`), tối ưu hóa lệnh `HEALTHCHECK` bằng script Python thuần (không cần cài thêm `curl`), và quản lý user bảo mật (`cvuser`) chứng tỏ bạn đã đầu tư nghiên cứu kỹ lưỡng.

Tuy nhiên, **lý do cốt lõi khiến Image của bạn vẫn có nguy cơ bị phình to (hoặc gặp lỗi tiềm ẩn khi chạy)** nằm ở cơ chế sao chép gói thư viện giữa hai Stage.

Dưới đây là 3 điểm "bẫy" trong Dockerfile này và cách xử lý triệt để:

---

## 🔎 Phân tích 3 điểm cần tinh chỉnh

### 1. Bẫy dung lượng và xung đột từ câu lệnh `COPY --from=builder /install /usr/local`

Khi bạn dùng lệnh `pip install --prefix=/install`, các gói thư viện Python sẽ được cài vào cấu trúc `/install/lib/python3.11/site-packages`.

- Khi bạn ném nó sang Stage 2 bằng lệnh `COPY --from=builder /install /usr/local`, bạn đang cố tình ghi đè toàn bộ các gói này vào thư mục hệ thống `/usr/local` của Stage Runtime.
- **Hệ lụy:** Cách này đôi khi vô tình kéo theo các file rác, file cấu hình build, hoặc làm mất đi các symlink tối ưu của Python gốc trong base image `slim`. Ngoài ra, nếu trong `requirements.txt` của bạn không ép cài bản **Torch CPU**, pip vẫn sẽ tải bản CUDA nặng 4-5GB về thư mục này, khiến kích thước không giảm được bao nhiêu.

### 2. Đường dẫn `PATH` chưa đồng bộ với User mới

Ở Stage Runtime, bạn cấu hình: `PATH="/home/cvuser/.local/bin:${PATH}"`.
Nhưng ở câu lệnh trước đó, bạn lại copy thư viện vào `/usr/local`. Các script thực thi (như `uvicorn`) lúc này sẽ nằm ở `/usr/local/bin` chứ không phải trong thư mục của `cvuser`. Điều này dễ dẫn đến lỗi hệ thống không tìm thấy lệnh `uvicorn` khi khởi chạy container dưới quyền user `cvuser`.

### 3. Giải pháp thay thế hoàn hảo: Dùng Môi trường ảo (`virtualenv`)

Thay vì dùng `--prefix` phức tạp và rủi ro, cách chuẩn công nghiệp (Production-ready) hiện nay là tạo một **Virtual Environment (`venv`)** ở Stage 1, cài hết thư viện vào đó, rồi copy nguyên thư mục `venv` sang Stage 2. Cách này cô lập 100% thư viện, cực kỳ sạch sẽ và dung lượng là tối thiểu.

---

## 🛠️ Dockerfile hoàn thiện & tối ưu tuyệt đối

Dưới đây là file của bạn sau khi đã được tinh chỉnh lại theo cơ chế `venv` an toàn và gọn nhẹ nhất:

```dockerfile
# syntax=docker/dockerfile:1.7

# ── STAGE 1: BUILDER ───────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Các thư viện cần để build gói hệ thống (nếu có)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# TỐI ƯU: Tạo một môi trường ảo riêng biệt tại /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Cài đặt dependencies (Ép cài Torch CPU nếu bạn deploy server không có GPU)
# Hãy đảm bảo cấu hình URL wheel CPU trong requirements.txt hoặc thêm trực tiếp vào đây nếu cần
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── STAGE 2: RUNTIME ───────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # TỐI ƯU: Trỏ thẳng PATH vào thư mục bin của môi trường ảo vừa copy sang
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Các thư viện runtime bắt buộc cho OpenCV hoạt động mượt mà
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# TỐI ƯU: Copy nguyên vẹn môi trường ảo sạch từ builder sang
COPY --from=builder /opt/venv /opt/venv

# Tạo user bảo mật cvuser
RUN useradd --create-home --uid 1001 --shell /bin/bash cvuser

# Copy source code ứng dụng và phân quyền cho cvuser
COPY --chown=cvuser:cvuser app/ ./app/
COPY --chown=cvuser:cvuser weights/ ./weights/

USER cvuser

EXPOSE 8000

# Giữ nguyên lệnh healthcheck thông minh bằng Python của bạn
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/cv/health', timeout=5)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop", "--no-access-log"]

```

---

## 💡 Đánh giá cuối cùng trước khi build

1. **Về cân nặng:** Nếu file `requirements.txt` của bạn đã cấu hình bản `cpu` (cho PyTorch/TensorFlow) và dùng `opencv-python-headless`, kết hợp với Dockerfile mới này, kích thước Image chắc chắn sẽ giảm từ **9GB xuống chỉ còn quanh mức ~1.5GB - 2GB** (đã bao gồm file weights vừa phải).
2. **Về độ an toàn:** File này cô lập hoàn toàn quyền chạy của `cvuser`, không đụng chạm vào thư mục gốc `/usr/local` của hệ thống, giúp tránh các lỗi phân quyền (Permission denied) khi chạy app.
