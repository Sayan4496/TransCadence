from pathlib import Path
from uuid import uuid4
import json

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.services.asr_service import transcribe_audio
from backend.app.services.audio_service import extract_audio
from backend.app.services.video_service import validate_video


router = APIRouter(
    prefix="/api",
    tags=["Video Upload"],
)


UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIO_DIR = Path("storage/audio")
TRANSCRIPT_DIR = Path("storage/transcripts")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/webm",
}


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):
    """
    Upload and validate a video.
    """

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only MP4 and WebM videos are supported.",
        )

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in {".mp4", ".webm"}:
        raise HTTPException(
            status_code=400,
            detail="Only .mp4 and .webm files are supported.",
        )

    job_id = str(uuid4())

    safe_filename = f"{job_id}{extension}"

    output_path = UPLOAD_DIR / safe_filename

    try:
        with output_path.open("wb") as buffer:

            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

    except Exception as exc:

        if output_path.exists():
            output_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save upload: {exc}",
        )

    try:

        metadata = validate_video(
            output_path
        )

    except ValueError as exc:

        output_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        output_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {exc}",
        )

    audio_path = AUDIO_DIR / f"{job_id}.wav"
    transcript_path = TRANSCRIPT_DIR / f"{job_id}.json"

    try:
        audio_path = await run_in_threadpool(
            extract_audio,
            output_path,
            job_id,
        )
    except Exception as exc:
        audio_path.unlink(missing_ok=True)
        transcript_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Audio extraction failed: {exc}",
        )

    try:
        transcript = await run_in_threadpool(
            transcribe_audio,
            audio_path,
        )

        with transcript_path.open("w", encoding="utf-8") as transcript_file:
            json.dump(
                transcript,
                transcript_file,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as exc:
        audio_path.unlink(missing_ok=True)
        transcript_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        )

    return {
        "job_id": job_id,
        "status": "transcribed",
        "filename": file.filename,
        "stored_filename": safe_filename,
        "duration": metadata["duration"],
        "metadata": metadata["metadata"],
        "transcript": transcript,
    }