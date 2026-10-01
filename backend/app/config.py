"""Backend-only environment configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_backend_environment() -> None:
    """Load backend/.env without overwriting variables already in the process."""
    load_dotenv(BACKEND_ENV_PATH, override=False)


def gemini_api_key_status() -> str:
    """Return only whether the server-side Gemini key is present."""
    return "present" if os.getenv("GEMINI_API_KEY") else "missing"


load_backend_environment()