"""Vision / Image understanding — encode, resize, and prepare images for multimodal LLMs."""

import base64
import io
import logging
import mimetypes
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Max image dimension (pixels) to keep context budget manageable
_MAX_DIMENSION = 2048
_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# Supported image formats
_SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}


def is_image_file(path: Path) -> bool:
    """Check if a file is a supported image format.

    Args:
        path: File path.

    Returns:
        True if the file is a supported image.
    """
    return path.suffix.lower() in _SUPPORTED_FORMATS


def encode_image(path: Path, max_dimension: int = _MAX_DIMENSION) -> dict[str, str]:
    """Encode an image file as base64 for LLM API calls.

    Resizes the image if it exceeds max_dimension to keep
    token costs down. Returns the data in the format expected
    by OpenAI/Anthropic/Google vision APIs.

    Args:
        path: Path to the image file.
        max_dimension: Max width/height in pixels.

    Returns:
        Dict with "type", "media_type", and "data" (base64 string).
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.stat().st_size > _MAX_FILE_SIZE:
        raise ValueError(f"Image too large: {path.stat().st_size / 1024 / 1024:.1f}MB (max {_MAX_FILE_SIZE / 1024 / 1024:.0f}MB)")

    # Detect MIME type
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"

    # Try to resize with Pillow
    try:
        from PIL import Image

        img = Image.open(path)
        width, height = img.size

        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            logger.debug("Resized image %s from %dx%d to %dx%d", path.name, width, height, *new_size)

        # Encode to bytes
        buffer = io.BytesIO()
        fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        img.save(buffer, format=fmt)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")

    except ImportError:
        # No Pillow — encode raw
        data = base64.b64encode(path.read_bytes()).decode("ascii")

    return {
        "type": "image",
        "media_type": mime_type,
        "data": data,
    }


def encode_image_bytes(raw: bytes, mime_type: str = "image/png") -> dict[str, str]:
    """Encode raw image bytes as base64.

    Args:
        raw: Raw image bytes.
        mime_type: MIME type of the image.

    Returns:
        Dict with "type", "media_type", and "data".
    """
    data = base64.b64encode(raw).decode("ascii")
    return {
        "type": "image",
        "media_type": mime_type,
        "data": data,
    }


def build_vision_message(
    text: str,
    images: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a multimodal message with text and images.

    Compatible with OpenAI's vision API format.

    Args:
        text: The text prompt.
        images: List of encoded image dicts from encode_image().

    Returns:
        Message dict with content array.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": text},
    ]

    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img['media_type']};base64,{img['data']}",
            },
        })

    return {"role": "user", "content": content}


def estimate_image_tokens(
    width: int,
    height: int,
    detail: str = "auto",
) -> int:
    """Estimate the token cost of an image for OpenAI models.

    Based on OpenAI's image token calculation:
    - low detail: 85 tokens
    - high detail: 170 * ceil(w/512) * ceil(h/512) + 85

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        detail: "low", "high", or "auto".

    Returns:
        Estimated token count.
    """
    import math

    if detail == "low":
        return 85

    # High detail
    tiles_w = math.ceil(width / 512)
    tiles_h = math.ceil(height / 512)
    return 170 * tiles_w * tiles_h + 85


def take_screenshot(output_path: Path) -> Optional[Path]:
    """Take a screenshot of the primary monitor.

    Requires the ``Pillow`` library with ``ImageGrab`` support.

    Args:
        output_path: Where to save the screenshot.

    Returns:
        Path to the saved screenshot, or None on failure.
    """
    try:
        from PIL import ImageGrab

        screenshot = ImageGrab.grab()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot.save(str(output_path))
        logger.info("Screenshot saved to %s", output_path)
        return output_path
    except ImportError:
        logger.error("Pillow is required for screenshots: pip install Pillow")
        return None
    except Exception as exc:
        logger.error("Screenshot failed: %s", exc)
        return None
