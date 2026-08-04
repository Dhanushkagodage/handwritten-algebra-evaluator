"""Client for ocr-service (Module 01).

Two deliberate choices:

1. We always call `/extract-pages`, never `/extract`, even for a single image.
   `/extract` derives its storage id purely from the uploaded filename
   (app.py:134), so two concurrent runs uploading `answer.jpg` would overwrite
   each other's file in data/raw between write and read. `/extract-pages`
   additionally suffixes an epoch, and we rename every part on the way out.

2. Health is `GET /` — ocr-service has no `/health` route.
"""
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.clients.base import parse_model, request_json, stage_timeout
from app.config import settings
from app.schemas.upstream_ocr import OcrExtractResponse, OcrMarkingSchemeResponse

#: (field_name, filename, bytes, content_type)
UploadPart = Tuple[str, str, bytes, str]

STAGE = "ocr"


def _form(question_text: str, ocr_mode: str, use_math_ocr: bool) -> Dict[str, str]:
    # ocr-service declares these as Form(...) fields; booleans go over the wire
    # as the strings FastAPI can coerce.
    return {
        "question_text": question_text or "",
        "ocr_mode": ocr_mode,
        "use_math_ocr": "true" if use_math_ocr else "false",
    }


async def extract_pages(
    client: httpx.AsyncClient,
    *,
    images: List[Tuple[str, bytes, str]],
    question_text: str = "",
    ocr_mode: Optional[str] = None,
    use_math_ocr: Optional[bool] = None,
) -> OcrExtractResponse:
    """POST /extract-pages with up to five ordered answer images."""
    files = [
        (f"image_{index}", (filename, content, content_type))
        for index, (filename, content, content_type) in enumerate(images, start=1)
    ]
    payload = await request_json(
        client,
        method="POST",
        url=f"{settings.ocr_service_url}/extract-pages",
        stage=STAGE,
        timeout=stage_timeout(settings.ocr_timeout_s),
        data=_form(
            question_text,
            ocr_mode or settings.default_ocr_mode,
            settings.default_use_math_ocr if use_math_ocr is None else use_math_ocr,
        ),
        files=files,
    )
    return parse_model(payload, OcrExtractResponse, stage=STAGE)


async def extract_marking_scheme(
    client: httpx.AsyncClient,
    *,
    image: Tuple[str, bytes, str],
    question_text: str = "",
) -> OcrMarkingSchemeResponse:
    """POST /extract-marking-scheme — the purpose-built scheme extractor.

    This replaces the frontend's old regex scrape of generic OCR text.
    """
    filename, content, content_type = image
    payload = await request_json(
        client,
        method="POST",
        url=f"{settings.ocr_service_url}/extract-marking-scheme",
        stage=STAGE,
        timeout=stage_timeout(settings.ocr_scheme_timeout_s),
        data={"question_text": question_text or ""},
        files=[("image", (filename, content, content_type))],
    )
    return parse_model(payload, OcrMarkingSchemeResponse, stage=STAGE)


async def ping(client: httpx.AsyncClient) -> Dict[str, Any]:
    response = await client.get(f"{settings.ocr_service_url}/", timeout=httpx.Timeout(3.0))
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else {}
