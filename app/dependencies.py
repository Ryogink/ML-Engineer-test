"""Composition root (AD-1). Constructs adapters once, wires via Depends."""

from contextlib import asynccontextmanager

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.ocr_service import OcrProvider
from app.services.ner_service import NerProvider


# Constructed once; rebuilt only if ocr_dpi/ner_model change (not supported in-session)
_ocr: OcrProvider | None = None
_ner: NerProvider | None = None


def get_ocr(settings: Settings = Depends(get_settings)) -> OcrProvider:
    global _ocr
    if _ocr is None:
        from app.adapters.tesseract_ocr import TesseractOcr
        _ocr = TesseractOcr(settings.ocr_dpi)
    return _ocr


def get_ner(settings: Settings = Depends(get_settings)) -> NerProvider:
    global _ner
    if _ner is None:
        from app.adapters.spacy_ner import SpacyNer
        _ner = SpacyNer(settings.ner_model)
    return _ner


@asynccontextmanager
async def lifespan(app):
    settings = get_settings()
    # Load spaCy model once at startup (fail fast if missing)
    get_ner(settings)
    app.state.settings = settings
    yield
