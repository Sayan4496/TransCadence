import json
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.app.services.translation_service import (
    GeminiAuthenticationError,
    GeminiModelConfigurationError,
    GeminiModelUnavailableError,
    GeminiNetworkError,
    GeminiQuotaError,
    GeminiResponseError,
    GeminiTimeoutError,
    LANGUAGE_NAMES,
    GeminiTranslationError,
    MissingGeminiAPIKeyError,
    translate_segments,
)


router = APIRouter(prefix="/api/jobs", tags=["Translation"])
TRANSCRIPT_DIR = Path("storage/transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


def _translation_failure(exc: GeminiTranslationError) -> tuple[int, str]:
    if isinstance(exc, MissingGeminiAPIKeyError):
        return 503, "Translation service is not configured on the server."
    if isinstance(exc, GeminiModelConfigurationError):
        return 503, "The configured Gemini model is not allowed by server policy."
    if isinstance(exc, GeminiAuthenticationError):
        return 502, "Gemini rejected the server's API credentials."
    if isinstance(exc, GeminiQuotaError):
        return 429, "Gemini quota or rate limit reached. Please retry later."
    if isinstance(exc, GeminiModelUnavailableError):
        return 503, "The configured Gemini model is unavailable to this API account."
    if isinstance(exc, GeminiTimeoutError):
        return 504, "Gemini timed out while translating. Please retry shortly."
    if isinstance(exc, GeminiNetworkError):
        return 502, "The Gemini API could not complete the translation request."
    if isinstance(exc, GeminiResponseError):
        return 502, "Gemini returned an empty, invalid, or non-Hindi response."
    return 502, "Gemini translation failed."


class TranslationRequest(BaseModel):
    target_language: str


@router.post("/{job_id}/translate")
async def translate_job(job_id: str, request: TranslationRequest):
    try:
        UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Transcribe the video first.",
        )

    if request.target_language not in LANGUAGE_NAMES:
        supported_languages = ", ".join(LANGUAGE_NAMES)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language. Supported languages: {supported_languages}.",
        )

    transcript_path = TRANSCRIPT_DIR / f"{job_id}.json"
    if not transcript_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Transcribe the video first.",
        )

    target_language = request.target_language
    target_language_name = LANGUAGE_NAMES[target_language]

    try:
        with transcript_path.open("r", encoding="utf-8") as transcript_file:
            transcript = json.load(transcript_file)

        if transcript.get("language") != "en":
            raise ValueError("Only English source transcripts are supported.")

        segments = await run_in_threadpool(
            translate_segments,
            transcript["segments"],
            target_language,
        )

        translated_transcript = {
            "job_id": job_id,
            "source_language": "en",
            "target_language": target_language,
            "target_language_name": target_language_name,
            "segments": segments,
        }
        translated_path = TRANSCRIPT_DIR / f"{job_id}_{target_language}.json"
        with translated_path.open("w", encoding="utf-8") as translated_file:
            json.dump(
                translated_transcript,
                translated_file,
                indent=2,
                ensure_ascii=False,
            )
    except GeminiTranslationError as exc:
        diagnostic = type(exc).__name__
        if isinstance(exc, GeminiNetworkError):
            diagnostic += (
                f" api_code={exc.api_code}"
                f" api_status={exc.api_status or 'unknown'}"
            )
        logger.error("Hindi translation failed (%s)", diagnostic)
        status_code, safe_detail = _translation_failure(exc)
        raise HTTPException(status_code=status_code, detail=safe_detail) from None
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Only English"):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.error("Unexpected Hindi translation failure (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error prevented translation.",
        ) from None

    return {
        "job_id": job_id,
        "status": "translated",
        "source_language": "en",
        "target_language": target_language,
        "target_language_name": target_language_name,
        "translation_file": str(translated_path),
        "segments": segments,
    }