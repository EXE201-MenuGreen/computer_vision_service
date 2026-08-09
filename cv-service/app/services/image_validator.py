import io
from PIL import Image
from fastapi import UploadFile, HTTPException
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def prepare_image_for_inference(image: Image.Image) -> bytes:
    """Resize an RGB image and encode it using the service-wide JPEG settings."""
    max_dimension = settings.image_max_dimension_px
    if max_dimension > 0 and max(image.size) > max_dimension:
        image = image.copy()
        image.thumbnail((max_dimension, max_dimension))

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=settings.image_jpeg_quality,
        optimize=True,
    )
    return buffer.getvalue()


async def validate_and_load_image(file: UploadFile) -> Image.Image:
    """
    Validate MIME type, file size, and load into PIL Image.
    Raises HTTPException on any validation failure.
    """
    # 1. MIME type check
    if file.content_type not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. "
                   f"Allowed: {settings.allowed_mime_types}",
        )

    # 2. Read bytes (stream once)
    raw = await file.read()

    # 3. Size check
    if len(raw) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Max {settings.max_image_size_mb} MB.",
        )

    # 4. Decode and convert to RGB
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        logger.warning("image_decode_failed", error=str(exc))
        raise HTTPException(status_code=400, detail="Cannot decode image file.")

    logger.info(
        "image_validated",
        size_bytes=len(raw),
        width=img.width,
        height=img.height,
    )
    return img
