"""
Central configuration for the receipt intake pipeline.

Everything is driven by environment variables (loaded from a local .env file,
which is gitignored) so no secrets ever end up in source control. See
.env.example for the full list of supported variables.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    anthropic_api_key: str
    anthropic_model: str
    openai_api_key: str
    openai_model: str
    ocr_quality_threshold: float


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider not in {"anthropic", "openai", "mock"}:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. Use 'anthropic', 'openai', or 'mock'."
        )

    return Settings(
        llm_provider=provider,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        ocr_quality_threshold=float(os.getenv("OCR_QUALITY_THRESHOLD", "0.55")),
    )
