import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.app.services.translation_service import (
    LANGUAGE_NAMES,
    translate_segments,
)


router = APIRouter(prefix="/api/jobs", tags=["Translation"])
TRANSCRIPT_DIR = Path("storage/transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


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
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Only English"):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(
            status_code=500,
            detail="Translation failed. Check HF_TOKEN and model access, then try again.",
        ) from exc

    return {
        "job_id": job_id,
        "status": "translated",
        "source_language": "en",
        "target_language": target_language,
        "target_language_name": target_language_name,
        "translation_file": str(translated_path),
        "segments": segments,
    }