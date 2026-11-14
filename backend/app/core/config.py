"""Application configuration."""
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_ROOT / ".env"

class Settings(BaseSettings):
    """Configuration loaded from environment variables."""
    # App Info
    APP_NAME: str = "MARA - Multi-Agent Research Assistant"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # CORS
    BACKEND_CORS_ORIGINS: Any = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://*.onrender.com"
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str) and v.strip() != "":
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return [
            "http://localhost:5173",
            "http://localhost:3000",
            "https://*.vercel.app",
            "https://*.onrender.com"
        ]

    # LLM Configuration
    # LLM Configuration (Azure OpenAI). Set these in backend/.env; nothing is hardcoded.
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_CHAT_API_KEY: str = ""
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = ""
    AZURE_OPENAI_CHAT_API_VERSION: str = ""
    # Optional pricing for cost estimates (USD per 1M tokens). Leave unset to report tokens only.
    LLM_INPUT_COST_PER_1M_TOKENS: float | None = None
    LLM_OUTPUT_COST_PER_1M_TOKENS: float | None = None

    @field_validator("LLM_INPUT_COST_PER_1M_TOKENS", "LLM_OUTPUT_COST_PER_1M_TOKENS", mode="before")
    @classmethod
    def blank_price_means_unset(cls, v: Any) -> Any:
        return None if isinstance(v, str) and not v.strip() else v
    MODEL_TEMPERATURE: float = 0.5
    MAX_TOKENS: int = 2000
    TIMEOUT: int = 60
    MAX_RETRIES: int = 3

    # Research Tools
    TAVILY_API_KEY: str | None = None

    # Workflow behavior
    ENABLE_HUMAN_REVIEW: bool = False
    HISTORY_LIMIT: int = 100

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()


def cors_options(origins: List[str]) -> Dict[str, Any]:
    """
    Split configured origins into exact matches and wildcard patterns.

    Starlette's `allow_origins` compares strings literally, so an entry like
    "https://*.vercel.app" never matches anything. Wildcards become an anchored
    `allow_origin_regex` where `*` matches exactly one subdomain label.
    """
    exact = [origin for origin in origins if "*" not in origin or origin == "*"]
    patterns = [
        re.escape(origin).replace(r"\*", r"[a-z0-9-]+")
        for origin in origins
        if "*" in origin and origin != "*"
    ]
    return {
        "allow_origins": exact,
        "allow_origin_regex": f"^(?:{'|'.join(patterns)})$" if patterns else None,
    }

settings = get_settings()