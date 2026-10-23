"""Application settings (AD-2)."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ocr_dpi: int = 300
    ner_model: str = "en_core_web_sm"
    max_upload_bytes: int = 20_000_000
    fuzzy_match_threshold: int = 90

    class Config:
        env_prefix = "DIS_"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()
