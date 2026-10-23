"""POST /api/extract — thin router delegating to services."""

import json

from fastapi import APIRouter, Depends, UploadFile, Form, File

from app.core.config import Settings, get_settings
from app.core.errors import external_guard
from app.services.ocr_service import validate_and_read
from app.dependencies import OcrProvider, get_ocr, get_ner
from app.models.schemas import ExtractionResponse, ExtractedName, BoundingBox
from app.services.bbox_service import find_name_bounding_boxes, pixels_to_points
from app.services.ner_service import extract_names

router = APIRouter()


@router.post("/extract", response_model=ExtractionResponse)
def extract(
    file: UploadFile = File(...),
    names_json: str = Form("[]"),
    settings: Settings = Depends(get_settings),
    ocr: OcrProvider = Depends(get_ocr),
    ner=Depends(get_ner),
):
    try:
        names = json.loads(names_json)
    except json.JSONDecodeError:
        names = []

    raw = validate_and_read(file, settings.max_upload_bytes)

    # Tracer: page 1 only. OCR failure surfaces as ExternalServiceError (502).
    page_index = 0
    text, pixel_words = external_guard(ocr.ocr_page)(raw, page_index, settings.ocr_dpi)

    detected = external_guard(extract_names)(text, ner)

    # Merge with any caller-provided names
    for nm in names:
        full = f"{nm.get('first_name', '')} {nm.get('last_name', '')}".strip()
        if full and full not in detected:
            detected.append(full)

    # Pixel→point conversion happens here, exactly once (AD-4)
    words = pixels_to_points(pixel_words, settings.ocr_dpi)

    results = []
    for name in detected:
        for box in find_name_bounding_boxes(name, words):
            results.append(ExtractedName(
                name=name,
                bounding_box=BoundingBox(
                    page_number=box["page"],
                    x=box["x"],
                    y=box["y"],
                    width=box["width"],
                    height=box["height"],
                ),
            ))

    return ExtractionResponse(extracted_names=results, fuzzy_matches=[])
